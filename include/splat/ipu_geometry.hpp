
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
  struct ivec4 operator-(ivec4 const &other) {
    return {x - other.x, y - other.y, z - other.z, w - other.w};
  }
  struct ivec4 operator*(float const &scalar) {
    return {x * scalar, y * scalar, z * scalar, w * scalar};
  }
};

typedef struct ivec4 ivec4;

struct ivec2 {
  float x;
  float y;
  struct ivec2 operator+(ivec2 const &other) const {
    return {x + other.x, y + other.y};
  }
  struct ivec2 operator-(ivec2 const &other) const {
    return {x - other.x, y - other.y};
  }
  struct ivec2 operator*(float const &scalar) const {
    return {x * scalar, y * scalar};
  }
  struct ivec2 operator/(float const &scalar) const {
    return {x / scalar, y / scalar};
  }
  float length() const {
    return sqrt(x * x + y * y);
  }
  static float manhattanDistance(ivec2 const &a, ivec2 const &b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
  }
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

enum direction {
  left,
  right,
  up,
  down
};

struct Bounds2f {
  Bounds2f(bool) {
    // Overload to skip default init. Used to preseve contents on references.
  }

  Bounds2f(const ivec2& _min, const ivec2& _max) : min(_min), max(_max) {}

  ivec2 centroid() const {
    return (max + min) * .5f;
  }

  ivec2 diagonal() const {
    return max - min;
  }

  // /// Extend the bounds to enclose another bounding box:
  // void operator += (const Bounds2f& other) {
  //   min.x = min(min.x, other.min.x);
  //   min.y = min(min.y, other.min.y);
  //   max.x = max(max.x, other.max.x);
  //   max.y = max(max.y, other.max.y);
  // }

  // /// Extend the bounds to enclose the specified point:
  // void operator += (const ivec2& v) {
  //   min.x = min(min.x, v.x);
  //   min.y = min(min.y, v.y);
  //   max.x = max(max.x, v.x);
  //   max.y = max(max.y, v.y);
  // }

  Bounds2f clip(const Bounds2f& fixedBound) const {
    ivec2 topleft = min;
    ivec2 bottomright = max;
    if (topleft.x < fixedBound.min.x) {
      topleft.x = fixedBound.min.x;
    }
    if (topleft.y < fixedBound.min.y) {
      topleft.y = fixedBound.min.y;
    }
    if (bottomright.x >= fixedBound.max.x) {
      bottomright.x = fixedBound.max.x;
    }
    if (bottomright.y >= fixedBound.max.y) {
      bottomright.y = fixedBound.max.y;
    }
    Bounds2f clipped = {topleft, bottomright};
    return clipped;
  }

  Bounds2f clip(const Bounds2f& fixedBound, directions& dirs) const {
    ivec2 topleft = min;
    ivec2 bottomright = max;
    if (topleft.x < fixedBound.min.x) {
      topleft.x = fixedBound.min.x;
      dirs.left = true;
    }
    if (topleft.y < fixedBound.min.y) {
      topleft.y = fixedBound.min.y;
      dirs.up = true;
    }
    if (bottomright.x >= fixedBound.max.x) {
      bottomright.x = fixedBound.max.x;
      dirs.right = true;
    }
    if (bottomright.y >= fixedBound.max.y) {
      bottomright.y = fixedBound.max.y;
      dirs.down = true;
    }
    Bounds2f clipped = {topleft, bottomright};
    return clipped;
  }

  bool contains(const ivec2& v) const {
    return v.x >= min.x && v.x < max.x && v.y >= min.y && v.y < max.y;
  }

  bool contains(const ivec4& v) const {
    return contains(ivec2{v.x, v.y});
  }

  void print() const {
    printf("tl : %f, %f  br: %f, %f", min.x, min.y, max.x, max.y);
  }

  ivec2 min;
  ivec2 max;
};

struct Primitive {
  ivec4 mean; // in world space
  ivec4 colour; // RGBA colour space
  unsigned gid;
  virtual Bounds2f getBoundingBox() const = 0;  
  virtual bool inside(float x, float y) const = 0;
};

struct square : Primitive {
  float radius = 10.f;

  Bounds2f getBoundingBox() const override {
    return Bounds2f({mean.x - radius, mean.y - radius}, {mean.x + radius, mean.y + radius});
  }

  bool inside(float x, float y) const override {
    return true;
  }

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

struct Gaussian : Primitive {
    float f_dc[3];  // first order spherical harmonics coeff (sRGB color space)
    float f_rest[45];  // more spherical harminics coeff
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

    Bounds2f getBoundingBox() const override {
      return Bounds2f({mean.x - 10.0f, mean.y - 10.0f}, {mean.x + 10.0f, mean.y + 10.0f});
    }

    bool inside(float x, float y) const override {
      return true;
    }
};

class Gaussian2D : public Primitive {
  public:
    ivec2 scale;
    ivec4 rot;  // local rotation of gaussian (real, i, j, k)

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

    // need to define IPU versions of these functions
    // since cos and sin are compiled differently?
    Bounds2f getBoundingBox() const override {
      auto c = glm::cos(rot.w);
      auto s = glm::sin(rot.w);
      glm::vec2 eigenvalues = {scale.x, scale.y};
      auto lambdas = (eigenvalues / 2.0f) * (eigenvalues / 2.0f);
      auto dxMax = glm::sqrt(lambdas.x * (c * c) + lambdas.y * (s * s));
      auto dyMax = glm::sqrt(lambdas.x * (s * s) + lambdas.y * (c * c));
      return Bounds2f({mean.x - dxMax, mean.y - dyMax}, {mean.x + dxMax, mean.y + dyMax});
    }

    bool inside(float x, float y) const override {
      // TODO: remove trig fns from per pixel test
      auto c = glm::cos(rot.w);
      auto s = glm::sin(rot.w);
      auto dd = (scale.x / 2) * (scale.x / 2);
      auto DD = (scale.y / 2) * (scale.y / 2);
      auto a = c * (x - mean.x) + s * (y - mean.y);
      auto b = s * (x - mean.x) - c * (y - mean.y);
      return (((a * a) / dd)  + ((b * b) / DD)) <= 1;
    }

    void print() {
      printf("mean: %f, %f, %f, %f ", mean.x, mean.y, mean.z, mean.w);
      printf("scale: %f, %f ", scale.x, scale.y);
      printf("rot: %f, %f, %f, %f ", rot.x, rot.y, rot.z, rot.w);
    }
};

#define GAUSSIAN_SIZE sizeof(Gaussian2D)

} // end of namespace splat