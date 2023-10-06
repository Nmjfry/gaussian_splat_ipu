// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

// Plain C++ Multi-Vertex to transform every 4x1 vector
// in an array by the same 4x4 transformation matrix:
class Transform4x4 : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float>> matrix;
  poplar::InOut<poplar::Vector<float>> vectors;

  // This implementation achieves approx 0.68 FLOPs/cycle:
  bool compute(unsigned workerId) {
    const auto startIndex = 4 * workerId;
    for (auto v = startIndex; v < vectors.size(); v += 4 * numWorkers()) {
      float x = vectors[v + 0];
      float y = vectors[v + 1];
      float z = vectors[v + 2];
      float w = vectors[v + 3];
      for (auto i = 0u; i < 4u; ++i) {
        vectors[v + i] = matrix[4 * i + 0] * x +
                         matrix[4 * i + 1] * y +
                         matrix[4 * i + 2] * z +
                         matrix[4 * i + 3] * w;
      }
    }
    return true;
  }
};
