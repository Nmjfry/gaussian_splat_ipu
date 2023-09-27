// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <ipu/options.hpp>
#include <ipu/ipu_utils.hpp>

#include <splat/geometry.hpp>
#include <splat/camera.hpp>
#include <splat/file_io.hpp>
#include <splat/serialise.hpp>

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.");
}

/// Apply modelview and projection transforms to points then accumulate results to an OpenCV image.
/// Returns the number of splatted points (the number of points that pass the image clip test).
std::uint32_t splatPoints(cv::Mat& image,
                          const glm::mat4& modelView, const glm::mat4& projection,
                          const splat::Viewport& viewport,
                          const splat::Points& pts,
                          std::uint8_t value=25) {
  std::uint32_t count = 0u;
  const auto mvp = projection * modelView;
  const auto colour = cv::Vec3b(value, value, value);

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto& v : pts) {
    // Project points to clip space:
    auto clipCoords = mvp * glm::vec4(v.p, 1.f);

    // Now convert to pixel coords:
    glm::vec2 windowCoords = viewport.clipSpaceToViewport(clipCoords);
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < image.rows && c < image.cols) {
      image.at<cv::Vec3b>(r, c) += colour;
      count += 1;
    }
  }

  return count;
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

  // Set up the modelling and projection transforms in an OpenGL compatible way:
  auto modelView = splat::lookAtBoundingBox(bb, glm::vec3(0.f , 1.f, 0.f), 1.f);

  // Splat all the points into an OpenCV image:
  cv::Mat image(720, 1280, CV_8UC3);
  image = 0.f;
  splat::Viewport viewport(0.f, 0.f, image.cols, image.rows);
  const float aspect = image.cols / (float)image.rows;

  // Transform the BB to camera/eye space:
  splat::Bounds3f bbInCamera(
    modelView * glm::vec4(bb.min, 1.f),
    modelView * glm::vec4(bb.max, 1.f)
  );
  ipu_utils::logger()->info("Point bounds (eye space): {}", bbInCamera);
  auto projection = splat::fitFrustumToBoundingBox(bbInCamera, glm::radians(40.0f), aspect);

  auto startTime = std::chrono::steady_clock::now();

  auto count = splatPoints(image, modelView, projection, viewport, pts);

  auto endTime = std::chrono::steady_clock::now();
  auto splatTimeSecs = std::chrono::duration<double>(endTime - startTime).count();

  ipu_utils::logger()->info("Total point count: {}", pts.size());
  ipu_utils::logger()->info("Splatted point count: {}", count);
  ipu_utils::logger()->info("Splat time: {} points/sec: {}", splatTimeSecs, pts.size()/splatTimeSecs);
  cv::imwrite("test.png", image);

  return EXIT_SUCCESS;
}
