// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/ext.hpp>

template <int Dim, typename Dtype, glm::qualifier defaultp>
std::ostream& operator<<(std::ostream& s, const glm::vec<Dim, Dtype, defaultp>& v) {
  s << glm::to_string(v);
  return s;
}

std::ostream& operator<<(std::ostream& s, const splat::Bounds3f& bb) {
  s << bb.min << " -> " << bb.max;
  return s;
}

#undef GLM_ENABLE_EXPERIMENTAL
