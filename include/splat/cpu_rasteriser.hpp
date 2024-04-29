// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>

#include <opencv2/imgproc.hpp>

#include <splat/geometry.hpp>
#include <splat/camera.hpp>
#include <splat/viewport.hpp>
#include <tileMapping/tile_config.hpp>

namespace splat {

/// Apply modelview and projection transforms to points:
void projectPoints(const splat::Points& in, const glm::mat4& projection, const glm::mat4& modelView,
                   std::vector<glm::vec4>& out);

/// Transform points from clip space into pixel coords and accumulate into an OpenCV image.
/// Returns the number of splatted points (the number of points that pass the image clip test).
std::uint32_t splatPoints(cv::Mat& image,
                          const std::vector<glm::vec4>& clipCoords,
                          const splat::Points& in,
                          const glm::mat4& mvp,
                          TiledFramebuffer& fb,
                          Viewport& vp,
                          std::uint8_t value=25);

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const std::vector<glm::vec4>& clipCoords,
                        TiledFramebuffer& fb,
                        Viewport& vp,
                        std::uint8_t value=25);

} // end of namespace splat
