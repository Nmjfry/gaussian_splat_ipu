// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/cpu_rasteriser.hpp>
#include <unordered_map>

namespace splat {

void projectPoints(const splat::Points& in, const glm::mat4& projection, const glm::mat4& modelView,
                   std::vector<glm::vec4>& out) {
  out.resize(in.size());
  const auto mvp = projection * modelView;

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto i = 0u; i < in.size(); ++i) {
    out[i] = mvp * glm::vec4(in[i].p, 1.f);
  }
}

std::uint32_t splatPoints(cv::Mat& image,
                          const std::vector<glm::vec4>& clipCoords,
                          const splat::Points& in,
                          const glm::mat4& mvp,
                          TiledFramebuffer& fb, 
                          Viewport& vp,
                          std::uint8_t value) {
  std::uint32_t count = 0u;
  // const auto colour = cv::Vec3b(0, 255, 0);
  // std::unordered_map<std::uint32_t, std::vector<glm::vec4>> copiedPoints;
  // const auto EXTENT = 5;

  // auto numPtsOnTile = clipCoords.size() / fb.numTiles;
  // #pragma omp parallel for schedule(static, 128) num_threads(32)
  // for (auto t = 0u; t < fb.numTiles; ++t) {
  //   auto [tlBound, brBound] = fb.getTileBounds(t);
  //   auto bufferStrip = std::vector<glm::vec4>(clipCoords.begin() + t * numPtsOnTile,
  //                                              clipCoords.begin() + (t + 1) * numPtsOnTile);

  //   for (auto i = 0u; i < 5; ++i) {

  //     // Convert from clip-space to pixel coords:
  //     glm::vec2 centre2D = vp.clipSpaceToViewport(bufferStrip[i]);

  //     // give point a square extent
  //     ivec2 topleft = {centre2D.x - (EXTENT / 2.0f), centre2D.y - (EXTENT / 2.0f)};
  //     ivec2 bottomright = {centre2D.x + (EXTENT / 2.0f), centre2D.y + (EXTENT / 2.0f)};

  //     // clip the square to the tile, return true 
  //     // if it needs to be copied to a different direction
  //     auto dirs = square::clip(tlBound, brBound, topleft, bottomright);

  //     // // Clip points to the image and splat:
  //     for (std::uint32_t i = topleft.x; i < bottomright.x; i++) {
  //       for (std::uint32_t j = topleft.y; j < bottomright.y; j++) {
  //         #pragma omp atomic update
  //         image.at<cv::Vec3b>(j, i) += colour;
  //         count += 1;
  //       }
  //     }

  //     if (dirs.right && t < fb.numTiles - 1) {
  //       copiedPoints[t + 1].push_back(glm::vec4(in[i].p, 1.f));
  //     } 
  //     if (dirs.left && t > 0) {
  //       copiedPoints[t - 1].push_back(glm::vec4(in[i].p, 1.f));
  //     } 
  //     if (dirs.up && t > fb.numTilesAcross) {
  //       copiedPoints[t - fb.numTilesAcross].push_back(glm::vec4(in[i].p, 1.f));
  //     }
  //     if (dirs.down && t + fb.numTilesAcross < fb.numTiles) {
  //       copiedPoints[t + fb.numTilesAcross].push_back(glm::vec4(in[i].p, 1.f));
  //     }

  //   }
  // }

  // const auto c2 = cv::Vec3b(0, 0, 255);
  // #pragma omp parallel for schedule(static, 128) num_threads(32)
  // for (const auto &[t, inPts] : copiedPoints) {
  //   auto bounds = fb.getTileBounds(t);

  //   for (auto i = 0u; i < inPts.size(); ++i) {
  //     // Convert from clip-space to pixel coords:
  //     glm::vec2 centre2D = vp.clipSpaceToViewport(mvp * inPts[i]);

  //     // give point a square extent
  //     ivec2 topleft = {centre2D.x - (EXTENT / 2.0f), centre2D.y - (EXTENT / 2.0f)};
  //     ivec2 bottomright = {centre2D.x + (EXTENT / 2.0f), centre2D.y + (EXTENT / 2.0f)};

  //     // clip the square to the tile, return true 
  //     // if it needs to be copied to a different direction
  //     auto dirs = square::clip(bounds.first, bounds.second, topleft, bottomright);

  //     // // Clip points to the image and splat:
  //     for (std::uint32_t i = topleft.x; i < bottomright.x; i++) {
  //       for (std::uint32_t j = topleft.y; j < bottomright.y; j++) {
  //         #pragma omp atomic update
  //         image.at<cv::Vec3b>(j, i) += c2;
  //         count += 1;
  //       }
  //     }

  //   }
  // }

  return count;
}

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const std::vector<glm::vec4>& clipCoords,
                        TiledFramebuffer& fb,
                        Viewport& vp,
                        std::uint8_t value) {
  std::uint32_t count = 0u;
  const auto colour = cv::Vec3b(value, value, value);

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto& c : counts) {
    c = 0u;
  }

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto& cs : clipCoords) {
    // Convert from clip-space to pixel coords:
    glm::vec2 windowCoords = vp.clipSpaceToViewport(cs);
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < fb.height && c < fb.width) {
      auto tidx = fb.pixCoordToTile(r, c);
      #pragma omp critical
      counts[tidx] += 1;
    }
  }
}

} // end of namespace splat
