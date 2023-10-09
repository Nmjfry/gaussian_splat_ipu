// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <cstdlib>

#include <opencv2/imgproc.hpp>

#include <splat/geometry.hpp>
#include <splat/camera.hpp>

namespace splat {

struct TiledFramebuffer {
  TiledFramebuffer(std::uint16_t w, std::uint16_t h, std::uint16_t tw, std::uint16_t th)
  :
    width(w), height(h),
    tileWidth(tw), tileHeight(th)
  {
    numTilesAcross = width / tileWidth;
    numTilesDown = height / tileHeight;
    numTiles = numTilesAcross * numTilesDown;
  }

  float pixCoordToTile(float row, float col) const {
    float r = std::nearbyint(row);
    float c = std::nearbyint(col);

    // Round to tile indices:
    float tileColIndex = std::floor(c / tileWidth);
    float tileRowIndex = std::floor(r / tileHeight);

    // Now flatten the tile indices:
    float tileIndex = (tileRowIndex * numTilesAcross) + tileColIndex;
    return tileIndex;
  }

  std::uint16_t width;
  std::uint16_t height;
  std::uint16_t tileWidth;
  std::uint16_t tileHeight;

  // Use floats for these so that indexing
  // calculations will be fast on IPU:
  float numTilesAcross;
  float numTilesDown;
  float numTiles;
};

/// Apply modelview and projection transforms to points:
void projectPoints(const splat::Points& in, const glm::mat4& projection, const glm::mat4& modelView,
                   std::vector<glm::vec4>& out);

/// Transform points from clip space into pixel coords and accumulate into an OpenCV image.
/// Returns the number of splatted points (the number of points that pass the image clip test).
std::uint32_t splatPoints(cv::Mat& image,
                          const std::vector<glm::vec4>& clipCoords,
                          const splat::Viewport& viewport,
                          std::uint8_t value=25);

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const TiledFramebuffer& fb,
                        const std::vector<glm::vec4>& clipCoords,
                        const splat::Viewport& viewport,
                        std::uint8_t value=25);

} // end of namespace splat
