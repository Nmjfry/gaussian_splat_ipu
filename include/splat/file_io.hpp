// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <splat/geometry.hpp>

#include <iostream>
#include <string>

namespace splat {

/// Very basic (and limited) funciton to load a pointcloud xyz text file:
Points loadXyz(std::istream&& s);

} // end of namespace splat
