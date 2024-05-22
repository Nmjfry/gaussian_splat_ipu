// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include <opencv2/highgui.hpp>

#include <ipu/options.hpp>
#include <ipu/ipu_utils.hpp>
#include <ipu/io_utils.hpp>
#include <splat/camera.hpp>

#include <splat/cpu_rasteriser.hpp>
#include <splat/ipu_rasteriser.hpp>
#include <splat/file_io.hpp>
#include <splat/serialise.hpp>

#include <splat/ipu_geometry.hpp>

#include <remote_ui/InterfaceServer.hpp>
#include <remote_ui/AsyncTask.hpp>

#include <pvti/pvti.hpp>

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
   "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.")
  ("ui-port", po::value<int>()->default_value(0), "Start a remote user-interface server on the specified port.")
  ("device", po::value<std::string>()->default_value("cpu"),
   "Choose the render device")
  ("no-amp", po::bool_switch()->default_value(true),
   "Disable use of optimised AMP codelets.");
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Points& pts, splat::TiledFramebuffer& fb, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts, fb, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Gaussians& pts, splat::TiledFramebuffer& fb, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts, fb, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
}

int main(int argc, char** argv) {
  pvti::TraceChannel traceChannel = {"splatter"};

  boost::program_options::options_description desc;
  addOptions(desc);
  boost::program_options::variables_map args;
  try {
    args = parseOptions(argc, argv, desc);
    setupLogging(args);
  } catch (const std::exception& e) {
    ipu_utils::logger()->info("Exiting after: {}.", e.what());
    return EXIT_FAILURE;
  }

  auto xyzFile = args["input"].as<std::string>();
  auto pts = splat::loadXyz(std::ifstream(xyzFile));
  splat::Bounds3f bb(pts);
  ipu_utils::logger()->info("Total point count: {}", pts.size());
  ipu_utils::logger()->info("Point bounds (world space): {}", bb);

  // Translate all points so the centroid is zero then negate the z-axis:
  {
    const auto bbCentre = bb.centroid();
    for (auto& v : pts) {
      v.p -= bbCentre;
      v.p.z = -v.p.z;
    }
    bb = splat::Bounds3f(pts);
  }

  // Splat all the points into an OpenCV image:
  auto imagePtr = std::make_unique<cv::Mat>(720, 1280, CV_8UC3);
  auto imagePtrBuffered = std::make_unique<cv::Mat>(imagePtr->rows, imagePtr->cols, CV_8UC3);
  const float aspect = imagePtr->cols / (float)imagePtr->rows;


  // Construct some tiled framebuffer histograms:
  splat::TiledFramebuffer fb(imagePtr->cols, imagePtr->rows, IPU_TILEWIDTH, IPU_TILEHEIGHT);
  auto pointCounts = std::vector<std::uint32_t>(fb.numTiles, 0u);

  auto num_pixels = imagePtr->rows * imagePtr->cols;
  auto pixels_per_tile = num_pixels / fb.numTiles;
  ipu_utils::logger()->info("Number of pixels in framebuffer: {}", num_pixels);
  ipu_utils::logger()->info("Number of tiles in framebuffer: {}", fb.numTiles);
  ipu_utils::logger()->info("Number of pixels per tile: {}", pixels_per_tile);

  float x = 719.f;
  float y = 1279.f;
  auto tileId = fb.pixCoordToTile(x, y);
  ipu_utils::logger()->info("Tile index test. Pix coord {}, {} -> tile id: {}", x, y, tileId);


  auto centre = bb.centroid();
  // make fb.numTiles copies of a 2D gaussian
  splat::Gaussians gsns;
  ipu_utils::logger()->info("Generating {} gaussians", pts.size());
  for (std::size_t i = 0; i < pts.size(); i++) {
    auto pt = pts[i].p;
    splat::Gaussian3D g;
    g.mean = {pt.x, pt.y, pt.z, 1.f};
    g.colour = {.4f, 0.f, .1f, 0.9f};
    g.scale = {.1f, .1f, .1f};
    g.gid = ((float) i)+1.f;
    gsns.push_back(g);
  }

  auto ipuSplatter = createIpuBuilder(gsns, fb, args["no-amp"].as<bool>());
  ipu_utils::GraphManager gm;
  gm.compileOrLoad(*ipuSplatter);

  // Setup a user interface server if requested:
  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  state.fov = glm::radians(40.f);
  state.device = args.at("device").as<std::string>();
  auto uiPort = args.at("ui-port").as<int>();
  if (uiPort) {
    uiServer.reset(new InterfaceServer(uiPort));
    uiServer->start();
    uiServer->initialiseVideoStream(imagePtr->cols, imagePtr->rows);
    uiServer->updateFov(state.fov);
  }

  // Set up the modelling and projection transforms in an OpenGL compatible way:
  auto modelView = splat::lookAtBoundingBox(bb, glm::vec3(0.f , 1.f, 0.f), 3.f);

  // Transform the BB to camera/eye space:
  splat::Bounds3f bbInCamera(
    modelView * glm::vec4(bb.min, 1.f),
    modelView * glm::vec4(bb.max, 1.f)
  );

  ipu_utils::logger()->info("Point bounds (eye space): {}", bbInCamera);
  auto projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);
  auto cameraTranslation = glm::mat4x4(1.f);

  ipuSplatter->updateModelViewProjection(projection * modelView);
  gm.prepareEngine();

  std::vector<glm::vec4> clipSpace;
  clipSpace.reserve(pts.size());
  splat::TiledFramebuffer cpufb(CPU_TILEWIDTH, CPU_TILEHEIGHT);
  splat::Viewport vp(0.f, 0.f, IMWIDTH, IMHEIGHT);

  // Video is encoded and sent in a separate thread:
  AsyncTask hostProcessing;
  auto uiUpdateFunc = [&]() {
    {
      pvti::Tracepoint scoped(&traceChannel, "ui_update");
      uiServer->sendHistogram(pointCounts);
      uiServer->sendPreviewImage(*imagePtrBuffered);
    }
    {
      pvti::Tracepoint scope(&traceChannel, "build_histogram");
      splat::buildTileHistogram(pointCounts, clipSpace, cpufb, vp);
    }
  };

  auto dynamicView = modelView;
  do {
    auto startTime = std::chrono::steady_clock::now();
    *imagePtr = 0;
    std::uint32_t count = 0u;

    if (state.device == "cpu") {
      pvti::Tracepoint scoped(&traceChannel, "mvp_transform_cpu");
      projectPoints(pts, projection, dynamicView, clipSpace);
      {
        pvti::Tracepoint scope(&traceChannel, "splatting_cpu");
        count = splat::splatPoints(*imagePtr, clipSpace, pts, projection * dynamicView, cpufb, vp);
      }
    } else if (state.device == "ipu") {
      pvti::Tracepoint scoped(&traceChannel, "mvp_transform_ipu");
      ipuSplatter->updateModelViewProjection(projection * dynamicView);
      gm.execute(*ipuSplatter);
      ipuSplatter->getFrameBuffer(*imagePtr);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto splatTimeSecs = std::chrono::duration<double>(endTime - startTime).count();

    if (uiServer) {
      hostProcessing.waitForCompletion();
      std::swap(imagePtr, imagePtrBuffered);
      hostProcessing.run(uiUpdateFunc);

      state = uiServer->consumeState();
      // Update projection:
      projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);
      // Update modelview:
      dynamicView = modelView * glm::rotate(glm::mat4(1.f), glm::radians(state.envRotationDegrees), glm::vec3(0.f, 1.f, 0.f));
      // dynamicView =dynamicView, glm::vec3(0.f, 0.f, -state.Z / 1000.f));
    } else {
      // Only log these if not in interactive mode:
      ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pts.size()/splatTimeSecs);
      ipu_utils::logger()->info("Splatted point count: {}", count);
    }

  } while (uiServer && state.stop == false);

  hostProcessing.waitForCompletion();

  cv::imwrite("test.png", *imagePtr);

  return EXIT_SUCCESS;
}
