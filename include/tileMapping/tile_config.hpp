#pragma once

#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/ipu_geometry.hpp>

#define IMWIDTH 1280.0f
#define IMHEIGHT 720.0f
#define TILES_ACCROSS 8.0f
#define TILES_DOWN 8.0f

#define CPU_TILEHEIGHT (IMHEIGHT / TILES_DOWN)
#define CPU_TILEWIDTH (IMWIDTH / TILES_ACCROSS)
// #define CPU_TILEHEIGHT 20.0f
// #define CPU_TILEWIDTH 32.0f
#define IPU_TILEHEIGHT (IMHEIGHT / TILES_DOWN)
#define IPU_TILEWIDTH  (IMWIDTH / TILES_ACCROSS)

namespace splat {

class TiledFramebuffer {
public:

  TiledFramebuffer(std::uint16_t tw, std::uint16_t th)
  : // TODO: template this on the image size and tile size
    width(IMWIDTH), height(IMHEIGHT),
    tileWidth(tw), tileHeight(th)
    // for now just assume the tile holds the whole image
  {
    numTilesAcross = width / tileWidth;
    numTilesDown = height / tileHeight;
    numTiles = numTilesAcross * numTilesDown;
  }

  TiledFramebuffer(std::uint16_t w, std::uint16_t h, std::uint16_t tw, std::uint16_t th) {
    width = w;
    height = h;
    tileWidth = tw;
    tileHeight = th;
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

  // Compute the tile's positition for a given tile index:
  Bounds2f getTileBounds(unsigned tid) const {
    auto div = floor(tid / numTilesAcross);
    auto mod = tid - div * numTilesAcross;
    ivec2 tl;
    ivec2 br;
    tl.x = floor(mod * tileWidth);
    tl.y = floor(div * tileHeight);
    br.x = tl.x + tileWidth;
    br.y = tl.y + tileHeight;
    auto bounds = Bounds2f(tl, br);
    // if (bounds.min.x < 0 || bounds.min.y < 0 || bounds.max.x > width || bounds.max.y > height) {
    //   printf("Warning!!! Bounds out of range: %f, %f, %f, %f\n", bounds.min.x, bounds.min.y, bounds.max.x, bounds.max.y);
    // }
    return bounds;
  }

  unsigned getNearbyTile(unsigned tid, const direction &recievedFrom) const {
    switch (recievedFrom) {
      case direction::left:
        return tid - 1;
      case direction::right:
        return tid + 1;
      case direction::up:
        return tid - numTilesAcross;
      case direction::down:
        return tid + numTilesAcross;
      default:
        return tid;
    }
  }

  static float manhattanDistance(ivec2 const &a, ivec2 const &b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
  }

  direction getBestDirection(ivec2 const &src, ivec2 const &dst) const {
    auto dist = manhattanDistance(src, dst);
    if (dist == 0) {
      return direction::none;
    }
    if (src.y < dst.y) {
      return direction::down;
    }
    if (src.y > dst.y) {
      return direction::up;
    }
    if (src.x < dst.x) {
      return direction::right;
    }
    if (src.x > dst.x) {
      return direction::left;
    }
    return direction::none;
  }

  bool isValidTile(unsigned tid) const {
    return tid < numTiles;
  }

  directions checkImageBoundaries(unsigned tid) const {
    directions dirs;
    auto [tl, br] = getTileBounds(tid);

    dirs.left = tl.x < 1;
    dirs.up = tl.y < 1;
    dirs.right = br.x > width - 1;
    dirs.down = br.y > height - 1;

    return dirs;
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
  unsigned tid;
};

} // end of namespace splat
