#pragma once

#include <glm/glm.hpp>

#define IMWIDTH 1280.0f
#define IMHEIGHT 720.0f
#define TILES_ACCROSS 40.0f
#define TILES_DOWN 36.0f

#define CPU_TILEHEIGHT (IMHEIGHT / TILES_DOWN)
#define CPU_TILEWIDTH (IMWIDTH / TILES_ACCROSS)
// #define CPU_TILEHEIGHT 20.0f
// #define CPU_TILEWIDTH 32.0f
#define IPU_TILEHEIGHT (IMHEIGHT / TILES_DOWN)
#define IPU_TILEWIDTH  (IMWIDTH / TILES_ACCROSS)

struct ivec4 {
  float x;
  float y;
  float z;
  float w;
  struct ivec4 operator+(ivec4 const &other) {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
  }

  void print() {
    printf("x: %f, y: %f, z: %f, w: %f\n", x, y, z, w);
  }
};

typedef struct ivec4 ivec4;

typedef struct {
  float x;
  float y;
} ivec2;

enum class Direction {
  UP,
  RIGHT,
  DOWN,
  LEFT,
  NUM_DIRS
};

typedef struct directions {
    bool up;
    bool right;
    bool down;
    bool left;
    bool keep;
    static const int NUM_DIRS = 4;
} directions;

class TiledFramebuffer {
public:

  TiledFramebuffer(std::uint16_t tw, std::uint16_t th)
  : // TODO: template this on the image size and tile size
    width(IMWIDTH), height(IMHEIGHT),
    tileWidth(tw), tileHeight(th), 
    // for now just assume the tile holds the whole image
    spec(0.f, 0.f, IMWIDTH, IMHEIGHT)
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
    spec = glm::vec4(0.f, 0.f, w, h);
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

  // Combine perspective division with viewport scaling:
  glm::vec2 clipSpaceToViewport(glm::vec4 cs) const {
    glm::vec2 vp(cs.x, cs.y);
    vp *= .5f / cs.w;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Converts from window coordinates to local tile coordinates:
  glm::vec2 viewportToTile(glm::vec2 windowCoords, unsigned tid) const {
    const auto [tl, br] = getTileBounds(tid);
    return glm::vec2(floor(windowCoords.x - tl.x), floor(windowCoords.y - tl.y));
  }

  // Converts from clip space to tile coordinates:
  glm::vec2 clipSpaceToTile(glm::vec4 cs, unsigned tid) const {
    glm::vec2 vp = clipSpaceToViewport(cs);
    return viewportToTile(vp, tid);
  }

  // Converts from normalised screen coords to the specified view window:
  glm::vec2 viewportTransform(glm::vec2 v) const {
    v.x *= spec[2];
    v.y *= spec[3];
    v.x += spec[0];
    v.y += spec[1];
    return v;
  }

  // Compute the tile's positition for a given tile index:
  std::pair<glm::vec2, glm::vec2> getTileBounds(unsigned tid) const {
    auto div = floor(tid / numTilesAcross);
    auto mod = tid - div * numTilesAcross;
    glm::vec2 tl;
    glm::vec2 br;
    tl.x = floor(mod * tileWidth);
    tl.y = floor(div * tileHeight);
    br.x = tl.x + tileWidth;
    br.y = tl.y + tileHeight;
    return std::make_pair(tl, br);
  }

  directions checkImageBoundaries(unsigned tid) {
    directions dirs;
    auto [tl, br] = getTileBounds(tid);

    dirs.left = tl.x < 1;
    dirs.up = tl.y < 1;
    dirs.right = br.x > width - 1;
    dirs.down = br.y > height - 1;

    return dirs;
  }

  glm::vec4 spec;
  
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

#define EXTENT 15.0f

struct square {
  ivec4 centre;
  ivec4 colour;
  unsigned gid;
  glm::vec2 topleft;
  glm::vec2 bottomright;

  square() {
    colour = {0, 1.0f, 0, 0};
  }

  void extend() {
    topleft = glm::vec2(centre.x - (EXTENT / 2.0f), centre.y - (EXTENT / 2.0f));
    bottomright = glm::vec2(centre.x + (EXTENT / 2.0f), centre.y + (EXTENT / 2.0f));
  }

  bool isOnTile(glm::vec2 pos, glm::vec2 tl, glm::vec2 br) {
    return pos.x >= tl.x && pos.x <= br.x && pos.y >= tl.y && pos.y <= br.y;
  }

  directions clip(std::pair<glm::vec2, glm::vec2> tileBounds) {
    directions dirs;
    auto [tl, br] = tileBounds;

    dirs.left = topleft.x < tl.x;
    dirs.up = topleft.y < tl.y;
    dirs.right = bottomright.x >= br.x;
    dirs.down = bottomright.y >= br.y;
    dirs.keep = isOnTile(topleft, tl, br) || isOnTile(bottomright, tl, br);

    if (dirs.left) {
      topleft.x = tl.x;
    }
    if (dirs.up) {
      topleft.y = tl.y;
    }
    if (dirs.right) {
      bottomright.x = br.x;
    }
    if (dirs.down) {
      bottomright.y = br.y;
    }
    return dirs;
  }
};
