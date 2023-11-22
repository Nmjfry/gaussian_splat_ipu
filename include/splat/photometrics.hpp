
#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <limits>


namespace splat {

struct Pixel {
  glm::vec3 rgb;
};

typedef std::vector<Pixel> Pixels;



} // end of namespace splat