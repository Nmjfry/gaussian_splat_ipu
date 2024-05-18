// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <glm/glm.hpp>

#include <splat/geometry.hpp>

namespace splat {

// Initialise the camera so it is slightly further than the bounding radius away
// down the +ve z-axis and looks at the centroid of the pointcloud:
glm::mat4 lookAtBoundingBox(const Bounds3f& bb, const glm::vec3& up, float scale);

} // end of namespace splat
