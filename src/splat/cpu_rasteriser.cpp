// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/cpu_rasteriser.hpp>

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
                          TiledFramebuffer& fb,
                          std::uint8_t value) {
  std::uint32_t count = 0u;
  const auto colour = cv::Vec3b(0, 255, 0);

  auto numPtsOnTile = clipCoords.size() / fb.numTiles;
  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto t = 0u; t < fb.numTiles; ++t) {
    auto [tl, br] = fb.getTileBounds(t);
    auto bufferStrip = std::vector<glm::vec4>(clipCoords.begin() + t * numPtsOnTile,
                                               clipCoords.begin() + (t + 1) * numPtsOnTile);
    for (auto i = 0u; i < bufferStrip.size(); ++i) {
      // Convert from clip-space to pixel coords:
      glm::vec2 tileCoords = fb.clipSpaceToTile(bufferStrip[i], t);
      auto sq = square(tileCoords);
      auto dirs = sq.clip(fb);

      // Clip points to the image and splat:
      for (std::uint32_t i = sq.topleft.x; i < sq.bottomright.x; i++) {
        for (std::uint32_t j = sq.topleft.y; j < sq.bottomright.y; j++) {
          #pragma omp atomic update
          image.at<cv::Vec3b>(j + tl.y, i + tl.x) += colour;
          count += 1;
        }
      }
    }
  }

  return count;
}

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const std::vector<glm::vec4>& clipCoords,
                        TiledFramebuffer& fb,
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
    glm::vec2 windowCoords = fb.clipSpaceToViewport(cs);
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
