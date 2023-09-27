// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/geometry.hpp>

#include <glm/gtc/matrix_transform.hpp>

namespace splat {

glm::mat4x4 fitFrustumToBoundingBox(const Bounds3f& bb, float fovRadians, float aspectRatio) {
  const float radius = glm::length(bb.diagonal()) * .5f;

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

} // end of namespace splat
