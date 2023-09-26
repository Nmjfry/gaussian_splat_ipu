// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <limits>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/program_options.hpp>

#include <splat/options.hpp>
#include <ipu/ipu_utils.hpp>

struct Point3f {
  glm::vec3 p;
  glm::vec3 rgb;
};

typedef std::vector<Point3f> Points;

// Object that holds the viewpoint specification and apply
// various viewport transforms:
struct Viewport {
  Viewport(float x, float y, float width, float height)
    : spec(x, y, width, height) {}

  // The input point should be in normalised device coords
  // (i.e. perspective division is already applied):
  glm::vec2 ndcToViewport(glm::vec4 ndc) {
    glm::vec2 vp(ndc.x, ndc.y);
    vp *= .5f;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Combine perspective division with viewport scaling:
  glm::vec2 clipSpaceToViewport(glm::vec4 cs) {
    glm::vec2 vp(cs.x, cs.y);
    vp *= .5f / cs.w;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Converts from normalised screen coords to the specified view window:
  glm::vec2 viewportTransform(glm::vec2 v) {
    v.x *= spec[2];
    v.y *= spec[3];
    v.x += spec[0];
    v.y += spec[1];
    return v;
  }

  glm::vec4 spec;
};

struct Bounds3f {
  Bounds3f(bool) {
    // Overload to skip default init. Used to preseve contents on references.
  }

  Bounds3f()
    : min(std::numeric_limits<float>::infinity()), max(-std::numeric_limits<float>::infinity()) {}
  Bounds3f(const glm::vec3& _min, const glm::vec3& _max) : min(_min), max(_max) {}
  Bounds3f(const Points& pts) : Bounds3f() {
    for (auto &vertex : pts) {
      *this += vertex.p;
    }
  }

  glm::vec3 centroid() const {
    return (max + min) * .5f;
  }

  glm::vec3 diagonal() const {
    return max - min;
  }

  void operator += (const Bounds3f& other) {
    min.x = std::min(min.x, other.min.x);
    min.y = std::min(min.y, other.min.y);
    min.z = std::min(min.z, other.min.z);
    max.x = std::max(max.x, other.max.x);
    max.y = std::max(max.y, other.max.y);
    max.z = std::max(max.z, other.max.z);
  }

  void operator += (const glm::vec3& v) {
    min.x = std::min(min.x, v.x);
    min.y = std::min(min.y, v.y);
    min.z = std::min(min.z, v.z);
    max.x = std::max(max.x, v.x);
    max.y = std::max(max.y, v.y);
    max.z = std::max(max.z, v.z);
  }

  glm::vec3 min;
  glm::vec3 max;
};

Points loadXyz(std::istream& s) {
  Points pts;
  pts.reserve(1000000);

  glm::vec3 ones(1.f, 1.f, 1.f);

  for (std::string line; std::getline(s, line); ) {
    std::stringstream ss(line);
    glm::vec3 p;
    ss >> p.x >> p.y >> p.z;
    pts.push_back({p, ones});
  }

  return pts;
}

void addOptions(boost::program_options::options_description& desc) {
  namespace po = boost::program_options;
  desc.add_options()
  ("help", "Show command help.")
  ("input,o", po::value<std::string>()->required(), "Input XYZ file.")
  ("log-level", po::value<std::string>()->default_value("info"),
  "Set the log level to one of the following: 'trace', 'debug', 'info', 'warn', 'err', 'critical', 'off'.");
}

// Return a projection matrix for the camera frustum that fits the given bounding box:
glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  auto radius = glm::length(bb.diagonal()) * .5f;

  // Position left, right, near and far planes from the radius:
  float nearPlane = radius / glm::tan(fovRadians);
  float farPlane = nearPlane + 20.f * radius;
  float halfWidth = radius * aspectRatio;
  float halfHeight = radius;

  float left = -halfWidth;
  float right = halfWidth;
  float bottom = -halfHeight;
  float top = halfHeight;

  return glm::frustum(left, right, bottom, top, nearPlane, farPlane);
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
  auto pts = loadXyz(fs);
  Bounds3f bb(pts);
  ipu_utils::logger()->info("Point bounds (world space): {} {} {} -> {} {} {}",
                            bb.min.x, bb.min.y, bb.min.z,
                            bb.max.x, bb.max.y, bb.max.z);

  // Set up the modelling and projection transforms in an OpenGL compatible way.

  // === Modelling ===

  // Initialise the camera so it is slightly further than the bounding radius away
  // down the +ve z-axis and looks at the centroid of the pointcloud:
  auto lookAtPoint = bb.centroid();
  auto radius = glm::length(bb.diagonal()) * .5f;
  auto cameraPosition = lookAtPoint - glm::vec3(0.f, 0.f, 1.f * radius);
  auto up = glm::vec3(0.f , 1.f, 0.f);
  auto modelView = glm::lookAt(cameraPosition, lookAtPoint, up);

  // === Projection ===

  // Splat all the points into an OpenCV image:
  cv::Mat image(720, 1280, CV_8UC3);
  image = 0.f;
  Viewport viewport(0.f, 0.f, image.cols, image.rows);
  const float aspect = image.cols / (float)image.rows;

  // Fit a perspective viewing frustum to the camera/eye space bounding box.

  // Transform the BB to eye space:
  auto bbMin = modelView * glm::vec4(bb.min, 1.f);
  auto bbMax = modelView * glm::vec4(bb.max, 1.f);
  Bounds3f bbInCamera(bbMin, bbMax);
  ipu_utils::logger()->info("Point bounds (eye space): {} {} {} -> {} {} {}",
                            bbInCamera.min.x, bbInCamera.min.y, bbInCamera.min.z,
                            bbInCamera.max.x, bbInCamera.max.y, bbInCamera.max.z);
  auto projection = fitFrustumToBoundingBox(bb, glm::radians(40.0f), aspect);

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
