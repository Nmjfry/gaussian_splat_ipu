// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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


// Test that GLM works on IPU:
class GlmMat4 : public poplar::MultiVertex {
public:
  bool compute(unsigned workerId) {
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
