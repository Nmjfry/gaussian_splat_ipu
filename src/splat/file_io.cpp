// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/file_io.hpp>

#include <sstream>
#include <algorithm>
#include <random>

namespace splat {

Points loadXyz(std::istream&& s) {
  Points pts;
  pts.reserve(1000000);

  glm::vec3 ones(1.f, 1.f, 1.f);

  for (std::string line; std::getline(s, line); ) {
    std::stringstream ss(line);
    glm::vec3 p;
    ss >> p.x >> p.y >> p.z;
    pts.push_back({p, ones});
  }

  // auto rng = std::default_random_engine {};
  // std::shuffle(std::begin(pts), std::end(pts), rng);

  return pts;
}

} // end of namespace splat
