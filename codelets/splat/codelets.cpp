// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

#define TILEHEIGHT 20.0f
#define TILEWIDTH 32.0f
#define IMWIDTH 1280.0f
#define IMHEIGHT 720.0f

struct square {
  glm::vec4 centre;
  glm::vec2 topleft;
  glm::vec2 bottomright;
  glm::vec4 colour = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

  square(glm::vec2 c) {
    topleft = glm::vec2(c.x - 5, c.y - 5);
    bottomright = glm::vec2(c.x + 5, c.y + 5);
    centre = glm::vec4(c.x, c.y, 0.0f, 1.0f);
  }
};


struct tileDims {
  glm::vec2 topleft;
  glm::vec2 bottomright;

  tileDims(int tid) {
    auto numBlocksWidth = IMWIDTH / TILEWIDTH;
    auto div = floor(tid / numBlocksWidth);
    auto mod = tid - div * numBlocksWidth;
    topleft.x = int(floor(mod * TILEWIDTH));
    topleft.y = int(floor(div * TILEHEIGHT));
    bottomright.x = topleft.x + int(TILEWIDTH);
    bottomright.y = topleft.y + int(TILEHEIGHT);
  }
};

// Object that holds the viewpoint specification and apply
// various viewport transforms:
struct Viewport {
  Viewport(float x, float y, float width, float height, unsigned tid)
    : spec(x, y, width, height), td(tid) {}

  // Combine perspective division with viewport scaling:
  glm::vec2 clipSpaceToViewport(glm::vec4 cs) const {
    glm::vec2 vp(cs.x, cs.y);
    vp *= .5f / cs.w;
    vp += .5f;
    return viewportTransform(vp);
  }

  // Converts from window coordinates to local tile coordinates:
  glm::vec2 viewportToTile(glm::vec2 windowCoords, struct tileDims td) const {
    return glm::vec2(floor(windowCoords.x - td.topleft.x), floor(windowCoords.y - td.topleft.y));
  }

  // Converts from clip space to tile coordinates:
  glm::vec2 clipSpaceToTile(glm::vec4 cs) const {
    glm::vec2 vp = clipSpaceToViewport(cs);
    return viewportToTile(vp, td);
  }

  // Converts from normalised screen coords to the specified view window:
  glm::vec2 viewportTransform(glm::vec2 v) const {
    v.x *= spec[2];
    v.y *= spec[3];
    v.x += spec[0];
    v.y += spec[1];
    return v;
  }

  glm::vec4 spec;
  struct tileDims td;
};

// Multi-Vertex to transform every 4x1 vector
// in an array by the same 4x4 transformation matrix.
// Uses the OpenGL Math (GLM) library for demonstration
// purposes (we do not expect this to be fast as GLM is
// not optimised for IPU yet).
//
// This is here as a reference to show what the
// accumulating matrix product (AMP) engine assembly
// vertices below are doing.
class Transform4x4 : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float>> matrix;
  poplar::Input<poplar::Vector<float>> vertsIn;
  poplar::Input<poplar::Vector<int>> tile_id;
  poplar::Input<poplar::Vector<float>> westIn; 
  poplar::Input<poplar::Vector<float>> eastOut;
  // instead of vertsOut we can have a vector of pixels 
  // corresponding to a pinned section of the framebuffer.
  poplar::Output<poplar::Vector<float>> localFb;
  unsigned bytesMoved = 0;


  int toByteBufferIndex(glm::vec2 pt) {
    return int(pt.x + pt.y * TILEWIDTH) * 4;
  } 

  bool isWithinTileBounds(glm::vec4 pt) {
    return pt.x >= 0 && pt.x < TILEWIDTH && pt.y >= 0 && pt.y < TILEHEIGHT;
  }
  bool isWithinTileBounds(glm::vec2 pt) {
    return pt.x >= 0 && pt.x < TILEWIDTH && pt.y >= 0 && pt.y < TILEHEIGHT;
  }

  void splat(struct square sq, poplar::Vector<float>& localFb) {
    for (auto i = sq.topleft.x; i < sq.bottomright.x; i++) {
      for (auto j = sq.topleft.y; j < sq.bottomright.y; j++) {
        auto rasterPixel = glm::vec2(i, j);
        if (!isWithinTileBounds(rasterPixel)) {
          // the structure (square) needs to be sent to a neighboring tile,
          // since the extent of the square is outside of this tile.
          continue;
        }
        auto index = toByteBufferIndex(rasterPixel);
        if (index < localFb.size() && index >= 0) {
          memcpy(&localFb[index], glm::value_ptr(sq.colour), sizeof(sq.colour));
        }
      }
    }
  }

  void rasterise(struct square sq, glm::vec4 unprojectedCentre) {
    if (isWithinTileBounds(sq.centre)) {
        // keep the center point on this tile
        splat(sq, localFb);
    } else {
      // determine which neighboring tile the point should be sent to
      if ((sq.centre.x >= TILEWIDTH || sq.centre.y >= TILEHEIGHT) && bytesMoved < eastOut.size() - sizeof(sq)) {
        void* ptr = (void*) &eastOut[bytesMoved];
        memcpy(ptr, glm::value_ptr(unprojectedCentre), sizeof(sq));
        bytesMoved+=sizeof(sq);
      }
    }
  }

  bool compute(unsigned workerId) {
    // Transpose because GLM storage order is column major:
    const auto m = glm::transpose(glm::make_mat4(&matrix[0]));
    struct Viewport viewport(0.f, 0.f, IMWIDTH, IMHEIGHT, tile_id[0]);

    for (auto i = 0; i < localFb.size(); i+=4) {
      auto black = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
      memcpy(&localFb[i], glm::value_ptr(black), sizeof(black));
    }

    for (auto i = 0; i < vertsIn.size(); i+=4) {
      auto upt = glm::make_vec4(&vertsIn[i]);
      auto pt = m * upt; 
      glm::vec2 tileCoords = viewport.clipSpaceToTile(pt);
      auto sq = square(tileCoords);
      rasterise(sq, upt);
    }

    for (auto i = 0; i < westIn.size(); i+=sizeof(struct square)) {
      auto upcentre = glm::make_vec4(&westIn[i]);
      auto colour = glm::make_vec4(&westIn[i+sizeof(glm::vec4)+sizeof(glm::vec2)+sizeof(glm::vec2)]);
      auto centre = m * upcentre; 
      glm::vec2 tileCoords = viewport.clipSpaceToTile(centre);
      auto sq = square(tileCoords);
      sq.colour.r = tile_id[0] * (1.0f / 1440);
      sq.colour.b = 1.0f;
      sq.colour.g = 0.0f;
      rasterise(sq, upcentre);
    }

    return true;
  }
};


#define CCCSLOAD 80

// Template class to calculate register values
// for common compute state registers:
template <unsigned N, unsigned M>
struct CWEI {
  static constexpr unsigned value = M + (N * 4);
};

// This vertex loads the AMP engine weight matrix. This has to be done
// in supervisor mode. The weight registers are automatically zeroed on
// entering supervisor mode so we only need to load the non-zero parts.
class LoadMatrix : public poplar::SupervisorVertex {
public:
  // Specify the alignment and that the matrix must be in interleaved memory:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 16, true>> matrix;

  bool compute() __attribute__((target("supervisor"))) {

    // Write the first address to load from into the $CCCSLOAD register:
    const auto loadStart = (unsigned)&matrix[0];

    // We want to load the 4x4 transform to upper left 4x4 block of the 16x16
    // common compute configuration registers $CWEI_N_M. Register indices are
    // calculated as index_of($CWEI_n_m) = m + n * 4.

    // Each ld128putcs instruction will read from the load address ($CCCSLOAD),
    // which must be in interleaved memory, and post increment it by 16 bytes:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_0_0 and CWEI_0_1:
    __builtin_ipu_ld128putcs(CWEI<0, 0>::value);
    // Load matrix slice [1, 0:3] to CWEI_1_0 and CWEI_1_1:
    __builtin_ipu_ld128putcs(CWEI<1, 0>::value);
    // Load matrix slice [2, 0:3] to CWEI_2_0 and CWEI_2_1:
    __builtin_ipu_ld128putcs(CWEI<2, 0>::value);
    // Load matrix slice [3, 0:3] to CWEI_3_0 and CWEI_3_1:
    __builtin_ipu_ld128putcs(CWEI<3, 0>::value);

    // Load the same 4x4 matrix into the lower right hand corner of weight matrix:
    __builtin_ipu_put(loadStart, CCCSLOAD);
    // Load matrix slice [0, 0:3] to CWEI_4_2 and CWEI_4_3:
    __builtin_ipu_ld128putcs(CWEI<4, 2>::value);
    // Load matrix slice [1, 0:3] to CWEI_5_2 and CWEI_5_3:
    __builtin_ipu_ld128putcs(CWEI<5, 2>::value);
    // Load matrix slice [2, 0:3] to CWEI_6_2 and CWEI_6_3:
    __builtin_ipu_ld128putcs(CWEI<6, 2>::value);
    // Load matrix slice [3, 0:3] to CWEI_7_2 and CWEI_7_3:
    __builtin_ipu_ld128putcs(CWEI<7, 2>::value);

    return true;
  }
};

// Small piece of ASM required to zero the AMP accumulator registers:
inline
void zeroFpAccumulators() {
  asm(R"(
    setzi $a0, 0x8
    uput $FP_CLR, $a0
  )"
  :
  :
  : "$a0");
}

// This vertex uses the Accumulating Matrix Product (AMP) engine to transform 4x1 vectors by
// a single fixed 4x4 matrix. I.e. it is an optimised verison of the Transform4x4 vertex above.
//
// Accumulating Matrix Product (AMP) engine.
// =========================================
//
// A matrix-vector product can be interpreted as taking a linear combination of the columns of
// the matrix. I.e. a matrix projects a vector into its "column space": the vector space spanned
// by its columns. This is exactly how the AMP engine works: it is a "column scaling" engine (a
// systolic array) where partially scaled columns are fed to the next unit in the array and
// results accumulated until the results drop out of the end of the pipeline.
//
// Each amp instruction (f32sisoamp is used here, but there are different variants) takes scalar
// elements from the input vector one by one and feeds that scalar to every engine. Each engine
// then multiples the scalar with elements from the weight matrix and passes the intermediate
// result to the next engine which will add the contribution of the next column to it.
//
// Execution is organised into phases. Different phases connect different weights to different
// engines. These connections are made such that each engine in a phase is responsible for scaling
// a part of the column of the weight matrix and accumulating the result to the accumulators. So
// each phase scales and accumulates one column from the weight matrix. Once all phases are complete
// the results are ready, but can only be extracted from the pipeline two elements at a time (and
// only on even phases for f32sisoamp).
//
// Additionally the AMP instruction can take a partial result which is also added to the scaled
// column. This allows executing larger matrix multiples by decomposing them into smaller blocks:
// each block can load a partial result, add to it, and eventually save result back to memory (which
// can be reloaded again later and so on). In our use case here, we do not need partial inputs so
// they are always zero. This also enables us to clear the accumulators ready for the next iteration.
// However, this does mean that in this application the available FLOPS relating to partials are not
// utilised, so we can not expect to reach the peak FLOP rate of the machine where the calculation
// does not actively load partials.
class Transform4x4_amp : public poplar::MultiVertex {
public:
  poplar::Input<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vertsIn;
  poplar::Output<poplar::Vector<float, poplar::VectorLayout::SPAN, 32, true>> vertsOut;

  bool compute(unsigned workerId) {
    zeroFpAccumulators();

    const unsigned startIndex = 8 * workerId;
    constexpr unsigned stride = 8 * numWorkers();
    constexpr unsigned step = 4 * numWorkers() - 3;
    // Ensure these pointers go in consecutive addresses:
    register const float* srcPtr asm("$m2") = &vertsIn[startIndex];
    register float* dstPtr asm("$m3") = &vertsOut[startIndex];
    const int span = vertsIn.size() - startIndex;
    unsigned iterations = span < 0 ? 0 : span / stride;

    asm (R"(
      .allow_optimizations

      # Fill (inject 8 elements):
      ld64step $a0:1, $mzero, %[loadAddr]+=, 1
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P0]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P2]
      {
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
      }
      {
        // Note we switch from using $a0:1 to using $a2:3 here to
        // free up more dual issue slots later:
        ld64step $a2:3, $mzero, %[loadAddr]+=, %[step]
        f32sisoamp $azeros, $a0, $azeros, %[TAMP_F32_E4_P4]
      }
      {
        // Pre-load first input pair before entering the loop.
        // (Note we switch back to loads into $a0:1 ready for the loop):
        ld64step $a0:1, $mzero, %[loadAddr]+=, 1
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      {
        // Optimised load/store instructions (ldst64pace below) require
        // triple packed addresses:
        tapack $m4:5, %[loadAddr], $mzero, %[storeAddr]
        f32sisoamp $azeros, $a2, $azeros, %[TAMP_F32_E4_P6]
      }

      # Main loop (inject 8 and retrieve 8 elements per iteration):
        .align 8
        {
          rpt %[iterations], 7
          f32sisoamp $azeros, $a3, $azeros, %[TAMP_F32_E4_P7] // This is not part of the loop
        }
        LOOP_START%=:
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P0]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P1]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P2]
        }
        {
          ldst64pace $a0:1, $a2:3, $m4:5+=, $mzero, 0b0000
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P3]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P4]
        }
        {
          // Use stride specification to jump the packed read pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0001
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
        }
        {
          nop
          f32sisoamp $a2:3, $a0, $azeros, %[TAMP_F32_E4_P6]
        }
        {
          // Use stride specification to jump the packed write pointer to the worker's next chunk:
          ldst64pace $a0:1, $a2:3, $m4:5+=, %[step], 0b0100 // At the end of the loop this is an over-read
          f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P7]
        }

      # Drain (retrieve and store the last 8 elements):
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P0]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P1]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P2]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $azero, $azeros, %[TAMP_F32_E4_P3]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P4]
      {
        st64pace $a2:3, $m4:5+=, $mzero, 0b00
        f32sisoamp $azeros, $a1, $azeros, %[TAMP_F32_E4_P5]
      }
      f32sisoamp $a2:3, $azero, $azeros, %[TAMP_F32_E4_P6]
      st64pace $a2:3, $m4:5+=, $mzero, 0b00
    )"
    : // outputs
    : [loadAddr] "r"(srcPtr), // inputs
      [storeAddr] "r"(dstPtr), // inputs
      [step] "r"(step),
      [iterations] "r"(iterations),
      [stride] "r"(stride),
      [TAMP_F32_E4_P0] "i"(TAMP_F32_E4_P0),
      [TAMP_F32_E4_P1] "i"(TAMP_F32_E4_P1),
      [TAMP_F32_E4_P2] "i"(TAMP_F32_E4_P2),
      [TAMP_F32_E4_P3] "i"(TAMP_F32_E4_P3),
      [TAMP_F32_E4_P4] "i"(TAMP_F32_E4_P4),
      [TAMP_F32_E4_P5] "i"(TAMP_F32_E4_P5),
      [TAMP_F32_E4_P6] "i"(TAMP_F32_E4_P6),
      [TAMP_F32_E4_P7] "i"(TAMP_F32_E4_P7)
    : "memory", "$m0", "$m4:5", "$a0:1", "$a2:3"); // clobbered

    return true;
  }
};
