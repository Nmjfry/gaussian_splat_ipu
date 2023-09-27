// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <limits>

namespace splat {

struct Point3f {
  glm::vec3 p;
  glm::vec3 rgb;
};

typedef std::vector<Point3f> Points;

struct Bounds3f {
  Bounds3f(bool) {
    // Overload to skip default init. Used to preseve contents on references.
  }

  Bounds3f()
    : min(std::numeric_limits<float>::infinity()), max(-std::numeric_limits<float>::infinity()) {}
  Bounds3f(const glm::vec3& _min, const glm::vec3& _max) : min(_min), max(_max) {}

  /// Initialises bounds to enclose all points in the list:
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

  /// Extend the bounds to enclose another bounding box:
  void operator += (const Bounds3f& other) {
    min.x = std::min(min.x, other.min.x);
    min.y = std::min(min.y, other.min.y);
    min.z = std::min(min.z, other.min.z);
    max.x = std::max(max.x, other.max.x);
    max.y = std::max(max.y, other.max.y);
    max.z = std::max(max.z, other.max.z);
  }

  /// Extend the bounds to enclose the specified point:
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

/// Return a projection matrix for the camera frustum that fits the given bounding box:
glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio);

} // end of namespace splat
