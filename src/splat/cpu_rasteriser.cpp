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
                          const splat::Viewport& viewport,
                          std::uint8_t value) {
  std::uint32_t count = 0u;
  const auto colour = cv::Vec3b(value, value, value);

  #pragma omp parallel for schedule(static, 128) num_threads(32)
  for (auto i = 0u; i < clipCoords.size(); ++i) {
    // Convert from clip-space to pixel coords:
    glm::vec2 windowCoords = viewport.clipSpaceToViewport(clipCoords[i]);
    std::uint32_t r = windowCoords.y;
    std::uint32_t c = windowCoords.x;

    // Clip points to the image and splat:
    if (r < image.rows && c < image.cols) {
      image.at<cv::Vec3b>(r, c) += colour;

      #pragma omp atomic update
      count += 1;
    }
  }

  return count;
}

void buildTileHistogram(std::vector<std::uint32_t>& counts,
                        const TiledFramebuffer& fb,
                        const std::vector<glm::vec4>& clipCoords,
                        const splat::Viewport& viewport,
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
    glm::vec2 windowCoords = viewport.clipSpaceToViewport(cs);
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
