#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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

struct ivec2 {
  float x;
  float y;
};

typedef struct ivec2 ivec2;

struct ivec3 {
  float x;
  float y;
  float z;
};

typedef struct ivec3 ivec3;

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
  std::pair<ivec2, ivec2> getTileBounds(unsigned tid) const {
    auto div = floor(tid / numTilesAcross);
    auto mod = tid - div * numTilesAcross;
    ivec2 tl;
    ivec2 br;
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

#define EXTENT 10.0f

struct square {
  ivec4 centre;
  ivec4 colour;
  unsigned gid;

  static bool isOnTile(ivec2 pos, ivec2 tlBound, ivec2 brBound) {
    return pos.x >= tlBound.x && pos.x <= brBound.x && pos.y >= tlBound.y && pos.y <= brBound.y;
  }

  static directions clip(ivec2 tlBound, ivec2 brBound, ivec2& topleft, ivec2& bottomright) {
    directions dirs;


    dirs.left = topleft.x < tlBound.x;
    dirs.up = topleft.y < tlBound.y;
    dirs.right = bottomright.x >= brBound.x;
    dirs.down = bottomright.y >= brBound.y;

    ivec2 topright = {bottomright.x, topleft.y};
    ivec2 bottomleft = {topleft.x, bottomright.y};
    dirs.keep = isOnTile(topleft, tlBound, brBound) || isOnTile(bottomright, tlBound, brBound) 
    || isOnTile(topright, tlBound, brBound) || isOnTile(bottomleft, tlBound, brBound);

    if (dirs.left) {
      topleft.x = tlBound.x;
    }
    if (dirs.up) {
      topleft.y = tlBound.y;
    }
    if (dirs.right) {
      bottomright.x = brBound.x;
    }
    if (dirs.down) {
      bottomright.y = brBound.y;
    }
    return dirs;
  }
};

struct Gaussian {
    float position[3];  // in world space
    float f_dc[3];  // first order spherical harmonics coeff (sRGB color space)
    float f_rest[45];  // more spherical harminics coeff
    ivec4 colour;
    float scale[3];
    float rot[4];  // local rotation of guassian (real, i, j, k)

    // convert from (scale, rot) into the gaussian covariance matrix in world space
    // See 3d Gaussian Splat paper for more info
    glm::mat3 ComputeCovMat() const
    {
        glm::quat q(rot[0], rot[1], rot[2], rot[3]);
        glm::mat3 R(glm::normalize(q));
        glm::mat3 S(glm::vec3(expf(scale[0]), 0.0f, 0.0f),
                    glm::vec3(0.0f, expf(scale[1]), 0.0f),
                    glm::vec3(0.0f, 0.0f, expf(scale[2])));
        return R * S * glm::transpose(S) * glm::transpose(R);
    }
};

struct Gaussian2D {
    ivec2 mean;  // in world space
    ivec2 scale;
    ivec4 rot;  // local rotation of gaussian (real, i, j, k)
    ivec4 colour; // RGBA color space

    // convert from (scale, rot) into the gaussian covariance matrix in world space
    // See 3d Gaussian Splat paper for more info
    glm::mat3 ComputeCovMat() const
    {
        glm::quat q(rot.x, rot.y, rot.z, rot.w);
        glm::mat3 R(glm::normalize(q));
        glm::mat3 S(glm::vec3(expf(scale.x), 0.0f, 0.0f),
                    glm::vec3(0.0f, expf(scale.y), 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.f));
        return R * S * glm::transpose(S) * glm::transpose(R);
    }
};
