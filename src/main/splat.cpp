// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include <opencv2/highgui.hpp>

#include <ipu/options.hpp>
#include <ipu/ipu_utils.hpp>
#include <ipu/io_utils.hpp>

#include <splat/cpu_rasteriser.hpp>
#include <splat/ipu_rasteriser.hpp>
#include <splat/file_io.hpp>
#include <splat/serialise.hpp>

#include <remote_ui/InterfaceServer.hpp>

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
   "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.")
  ("ui-port", po::value<int>()->default_value(0), "Start a remote user-interface server on the specified port.");
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Points& pts) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
}

int main(int argc, char** argv) {
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
  cv::Mat image(720, 1280, CV_8UC3);
  splat::Viewport viewport(0.f, 0.f, image.cols, image.rows);
  const float aspect = image.cols / (float)image.rows;

  // Construct some tiled framebuffer histograms:
  splat::TiledFramebuffer fb(image.cols, image.rows, 40, 16);
  auto pointCounts = std::vector<std::uint32_t>(fb.numTiles, 0u);

  ipu_utils::logger()->info("Number of tiles in framebuffer: {}", fb.numTiles);
  float x = 719.f;
  float y = 1279.f;
  auto tileId = fb.pixCoordToTile(x, y);
  ipu_utils::logger()->info("Tile index test. Pix coord {}, {} -> tile id: {}", x, y, tileId);

  auto ipuSplatter = createIpuBuilder(pts);

  ipu_utils::GraphManager gm;
  gm.compileOrLoad(*ipuSplatter);

  // Setup a user interface server if requested:
  std::unique_ptr<InterfaceServer> uiServer;
  InterfaceServer::State state;
  state.fov = glm::radians(40.f);
  auto uiPort = args.at("ui-port").as<int>();
  if (uiPort) {
    uiServer.reset(new InterfaceServer(uiPort));
    uiServer->start();
    uiServer->initialiseVideoStream(image.cols, image.rows);
    uiServer->updateFov(state.fov);
  }

  // Set up the modelling and projection transforms in an OpenGL compatible way:
  auto modelView = splat::lookAtBoundingBox(bb, glm::vec3(0.f , 1.f, 0.f), 1.f);

  // Transform the BB to camera/eye space:
  splat::Bounds3f bbInCamera(
    modelView * glm::vec4(bb.min, 1.f),
    modelView * glm::vec4(bb.max, 1.f)
  );
  ipu_utils::logger()->info("Point bounds (eye space): {}", bbInCamera);
  auto projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);

  ipuSplatter->updateModelViewProjection(modelView * projection);
  gm.prepareEngine();

  auto dynamicView = modelView;
  std::vector<glm::vec4> clipSpace;
  do {
    auto startTime = std::chrono::steady_clock::now();
    image = 0;

    if (state.device == "cpu") {
      projectPoints(pts, projection, dynamicView, clipSpace);
    } else if (state.device == "ipu") {
      ipuSplatter->updateModelViewProjection(projection * dynamicView);
      gm.execute(*ipuSplatter);
      ipuSplatter->getProjectedPoints(clipSpace);
    }

    buildTileHistogram(pointCounts, fb, clipSpace, viewport);
    auto count = splatPoints(image, clipSpace, viewport);

    auto endTime = std::chrono::steady_clock::now();
    auto splatTimeSecs = std::chrono::duration<double>(endTime - startTime).count();
    if (uiServer) {
      state = uiServer->consumeState();
      uiServer->sendHistogram(pointCounts);
      uiServer->sendPreviewImage(image);
      // Update projection:
      projection = splat::fitFrustumToBoundingBox(bbInCamera, state.fov, aspect);
      // Update modelview:
      dynamicView = modelView * glm::rotate(glm::mat4(1.f), glm::radians(state.envRotationDegrees), glm::vec3(0.f, 1.f, 0.f));
    } else {
      // Only log these if not in interactive mode:
      ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pts.size()/splatTimeSecs);
      ipu_utils::logger()->info("Total point count: {}", pts.size());
      ipu_utils::logger()->info("Splatted point count: {}", count);
    }
  } while (uiServer && state.stop == false);

  cv::imwrite("test.png", image);

  auto count = 0u;
  auto tile = 0u;
  auto emptyTiles = 0u;
  std::fstream of("out.txt");
  for (auto& c : pointCounts) {
    of << tile << " " << c << "\n";
    tile += 1;
    count += c;
    if (c == 0) {
      emptyTiles += 1;
    }
  }

  ipu_utils::logger()->info("Histogram point count: {}", count);
  ipu_utils::logger()->info("Tiles with no work to do: {}", emptyTiles);

  return EXIT_SUCCESS;
}
