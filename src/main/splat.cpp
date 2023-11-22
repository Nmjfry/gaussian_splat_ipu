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
#include <splat/photometrics.hpp>

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
  ("no-amp", po::bool_switch()->default_value(false),
   "Disable use of optimised AMP codelets.");
}

std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Points& pts, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pts, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
} 

// (nfry) overload function for pixels
// TODO (nfry): perhaps make Points and Pixels inherit from "primitive" interface.
std::unique_ptr<splat::IpuSplatter> createIpuBuilder(const splat::Pixels& pxs, bool useAMP) {
  using namespace poplar;

  ipu_utils::RuntimeConfig defaultConfig {
    1, 1, // numIpus, numReplicas
    "ipu_splatter", // exeName
    false, false, false, // useIpuModel, saveExe, loadExe
    false, true // compileOnly, deferredAttach
  };

  auto ipuSplatter = std::make_unique<splat::IpuSplatter>(pxs, useAMP);
  ipuSplatter->setRuntimeConfig(defaultConfig);
  return ipuSplatter;
} 

splat::Pixels gatherPixels(cv::Mat &image) {
  splat::Pixels pxs;
  pxs.reserve(image.cols * image.rows * 4);

  for (int row = 0; row < image.rows; row++) {
    for (int col = 0; col < image.cols; col++) {
        splat::Pixel pix = image.at<splat::Pixel>(col, row);
        pxs.push_back(pix);
    }
  }

  return pxs;
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


  // ###############################################################
  // ###############################################################
  // ###############################################################

  // TODO (nfry): get image buffer, create openCV image to process:

  // create an OpenCV image to contain pixels from camera:
  auto imagePtr = std::make_unique<cv::Mat>(2160,3840,CV_8UC3,15360);
  auto imagePtrBuffered = std::make_unique<cv::Mat>(imagePtr->rows, imagePtr->cols, CV_8UC3,15360);
  splat::Viewport viewport(0.f, 0.f, imagePtr->cols, imagePtr->rows);

  // Construct some tiled framebuffer histograms:
  splat::TiledFramebuffer fb(imagePtr->cols, imagePtr->rows, 90, 64);
  auto pointCounts = std::vector<std::uint32_t>(fb.numTiles, 0u);


  // TODO (nfry): edit hard coded image dims:

  ipu_utils::logger()->info("Number of tiles in framebuffer: {}", fb.numTiles);
  float x = 2159.f;
  float y = 3839.f;
  auto tileId = fb.pixCoordToTile(x, y);
  ipu_utils::logger()->info("Tile index test. Pix coord {}, {} -> tile id: {}", x, y, tileId);

  // TODO (nfry): Instead of building with pts, use pixels from image rgba:
  // will need to modify codelets otherwise some wierd things will happen to the pixels. 
  auto pxs = gatherPixels(*imagePtr);
  ipu_utils::logger()->info("Number of pixels in the image: {}", pxs.size());
  auto ipuSplatter = createIpuBuilder(pxs, args["no-amp"].as<bool>());
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
  }

  gm.prepareEngine();

  std::vector<glm::vec4> transformedPixels;
  transformedPixels.reserve(pxs.size());

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
      buildTileHistogram(pointCounts, fb, transformedPixels, viewport);
    }
  };

  do {
    auto startTime = std::chrono::steady_clock::now();
    *imagePtr = 0;
    uiServer->getCameraFrame(*imagePtr);

    if (state.device == "cpu") {
      // pvti::Tracepoint scoped(&traceChannel, "mvp_transform_cpu");
      // projectPoints(pts, projection, dynamicView, clipSpace);
    } else if (state.device == "ipu") {
      pvti::Tracepoint scoped(&traceChannel, "mvp_transform_ipu");

      // TODO (nfry): instead of changing projection, decode a new frame and
      // write the pixel values to the frame buffer
      // ipuSplatter->updatePixels(*imagePtr);

      gm.execute(*ipuSplatter);

      // TODO (nfry): instead of loading projected points into clipSpace,
      // load transformed pixels. 
      ipuSplatter->getProjectedPoints(transformedPixels);
    }

    unsigned count = 0u;
    {
      count = splat::writeTransformedPixels(*imagePtr, transformedPixels);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto splatTimeSecs = std::chrono::duration<double>(endTime - startTime).count();

    if (uiServer) {

      hostProcessing.waitForCompletion();
      std::swap(imagePtr, imagePtrBuffered);
      
      hostProcessing.run(uiUpdateFunc);
      state = uiServer->consumeState();

    } else {
      // Only log these if not in interactive mode:
      ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pxs.size()/splatTimeSecs);
      ipu_utils::logger()->info("Splatted point count: {}", count);
    }
  } while (uiServer && state.stop == false);

  hostProcessing.waitForCompletion();

  cv::imwrite("test.png", *imagePtr);

  return EXIT_SUCCESS;
}
