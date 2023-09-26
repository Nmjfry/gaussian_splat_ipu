// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#define CHECK_EQUAL(a,b) \
do { \
  if ((a) != (b)) { \
    printf("\n"); assert(a == b); \
  } \
} while(0)

#define CHECK_LE(a,b) \
do { \
  if ((a) > (b)) { \
    printf("\n"); assert(a > b); \
  } \
} while(0)

// Test that GLM works on IPU:
class GlmMat4 : public poplar::Vertex {
public:
  bool compute() {
    float m[] = {1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16};
    float v[] = {2, 4, 6, 8};
    const auto mgl = glm::transpose(glm::make_mat4(m));
    auto vgl = glm::make_vec4(v);
    vgl = mgl * vgl;
    CHECK_EQUAL(vgl.x, 60);
    CHECK_EQUAL(vgl.y, 140);
    CHECK_EQUAL(vgl.z, 220);
    CHECK_EQUAL(vgl.w, 300);
    return true;
  }
};

class GlmTransform : public poplar::Vertex {
public:
  bool compute() {
    glm::vec4 position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 view = glm::lookAt(glm::vec3(10.f, 10.f, 10.f),
                                 glm::vec3(0.f, 0.f, 0.0),
                                 glm::vec3(0.f, 1.f, 0.0));

    glm::vec4 transformed = view * position;
    CHECK_EQUAL(transformed.x, 0.f);
    CHECK_EQUAL(transformed.y, 0.f);
    CHECK_EQUAL(transformed.w, 1.f);
    auto zAbsErr = abs(-sqrt(300.f) - transformed.z );
    CHECK_LE(zAbsErr, 0.00001f);
    return true;
  }
};
