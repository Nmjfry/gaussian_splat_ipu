// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include <splat/geometry.hpp>

namespace splat {

// Object that holds the viewpoint specification and apply
// various viewport transforms:
struct Viewport {
  Viewport(float x, float y, float width, float height)
    : spec(x, y, width, height) {}

  // The input point should be in normalised device coords
  // (i.e. perspective division is already applied):
  glm::vec2 ndcToViewport(glm::vec4 ndc) const {
    glm::vec2 vp(ndc.x, ndc.y);
    vp *= .5f;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Combine perspective division with viewport scaling:
  glm::vec2 clipSpaceToViewport(glm::vec4 cs) const {
    glm::vec2 vp(cs.x, cs.y);
    vp *= .5f / cs.w;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Converts from normalised screen coords to the specified view window:
  glm::vec2 viewportTransform(glm::vec2 v) const {
    v.x *= spec[2];
    v.y *= spec[3];
    v.x += spec[0];
    v.y += spec[1];
    return v;
  }

  glm::vec4 spec;
};

// Initialise the camera so it is slightly further than the bounding radius away
// down the +ve z-axis and looks at the centroid of the pointcloud:
glm::mat4 lookAtBoundingBox(const Bounds3f& bb, const glm::vec3& up, float scale);

} // end of namespace splat
