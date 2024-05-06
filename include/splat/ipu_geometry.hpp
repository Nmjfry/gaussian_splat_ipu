
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace splat {

struct ivec4 {
  float x;
  float y;
  float z;
  float w;
  struct ivec4 operator+(ivec4 const &other) {
    return {x + other.x, y + other.y, z + other.z, w + other.w};
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

typedef struct directions {
    bool up;
    bool right;
    bool down;
    bool left;
    bool keep;
    static const int NUM_DIRS = 4;
} directions;

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

} // end of namespace splat