// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ipu/options.hpp>
#include <ipu/ipu_utils.hpp>

#include <splat/geometry.hpp>
#include <splat/camera.hpp>
#include <splat/file_io.hpp>

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.");
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
  auto fs = std::ifstream(xyzFile);
  auto pts = splat::loadXyz(fs);
  splat::Bounds3f bb(pts);
  ipu_utils::logger()->info("Point bounds (world space): {} {} {} -> {} {} {}",
                            bb.min.x, bb.min.y, bb.min.z,
                            bb.max.x, bb.max.y, bb.max.z);

  // Set up the modelling and projection transforms in an OpenGL compatible way:
  auto modelView = splat::lookAtBoundingBox(bb, glm::vec3(0.f , 1.f, 0.f), 1.f);

  // Splat all the points into an OpenCV image:
  cv::Mat image(720, 1280, CV_8UC3);
  image = 0.f;
  splat::Viewport viewport(0.f, 0.f, image.cols, image.rows);
  const float aspect = image.cols / (float)image.rows;

  // Fit a perspective viewing frustum to the camera/eye space bounding box.

  // Transform the BB to eye space:
  auto bbMin = modelView * glm::vec4(bb.min, 1.f);
  auto bbMax = modelView * glm::vec4(bb.max, 1.f);
  splat::Bounds3f bbInCamera(bbMin, bbMax);
  ipu_utils::logger()->info("Point bounds (eye space): {} {} {} -> {} {} {}",
                            bbInCamera.min.x, bbInCamera.min.y, bbInCamera.min.z,
                            bbInCamera.max.x, bbInCamera.max.y, bbInCamera.max.z);
  auto projection = splat::fitFrustumToBoundingBox(bb, glm::radians(40.0f), aspect);

  auto count = 0u;
  for (auto& v : pts) {
    // Project points to clip space:
    auto clipCoords = projection * modelView * glm::vec4(v.p, 1.f);

    // Finally convert to pixel coords:
    glm::vec2 windowCoords = viewport.clipSpaceToViewport(clipCoords);
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < image.rows && c < image.cols) {
      image.at<cv::Vec3b>(r, c) += cv::Vec3b(25, 25, 25);
      count += 1;
    }
  }

  ipu_utils::logger()->info("Total point count: {}", pts.size());
  ipu_utils::logger()->info("Splatted point count: {}", count);
  cv::imwrite("test.png", image);

  return EXIT_SUCCESS;
}
