// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/tileMapping/tile_config.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/viewport.hpp>

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

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
  poplar::Output<poplar::Vector<float>> localFb;
  poplar::Input<poplar::Vector<int>> tile_id;

  poplar::Input<poplar::Vector<float>> rightIn;
  poplar::Output<poplar::Vector<float>> rightOut;
  unsigned squaresSentRight = 0;

  poplar::Input<poplar::Vector<float>> leftIn; 
  poplar::Output<poplar::Vector<float>> leftOut;
  unsigned squaresSentLeft = 0;

  poplar::Input<poplar::Vector<float>> upIn;
  poplar::Output<poplar::Vector<float>> upOut;
  unsigned squaresSentUp = 0;

  poplar::Input<poplar::Vector<float>> downIn;
  poplar::Output<poplar::Vector<float>> downOut;
  unsigned squaresSentDown = 0;

  poplar::Input<poplar::Vector<float>> squares;

  unsigned gaussiansInitialised = 0;

  unsigned toByteBufferIndex(float x, float y) {
    return unsigned(x + y * IPU_TILEWIDTH) * 4;
  } 

  void setPixel(float x, float y, ivec4 &colour) {
    ivec4 pixel;
    unsigned idx = toByteBufferIndex(x, y);
    memcpy(&pixel, &localFb[idx], sizeof(pixel));
    pixel = pixel + colour;
    memcpy(&localFb[idx], &pixel, sizeof(pixel));
  }

  void splat(ivec4 colour, ivec2 tl, ivec2 br) {
    for (auto i = tl.x; i < br.x; i++) {
      for (auto j = tl.y; j < br.y; j++) {
        setPixel(i, j, colour);
      }
    }
  }

  void viewspaceToTile(ivec2& pt, ivec2 tlBound) {
    pt.x = floor(pt.x - tlBound.x);
    pt.y = floor(pt.y - tlBound.y);
  }

  bool insert(poplar::Vector<float> &buffer, unsigned idx, struct square& sq) {
    if (idx + sizeof(square) > buffer.size()) {
      return false;
    }
    memcpy(&buffer[idx], &sq, sizeof(sq.centre));
    memcpy(&buffer[idx+sizeof(sq.centre)], &sq.colour, sizeof(sq.colour));
    memcpy(&buffer[idx+sizeof(sq.centre)+sizeof(sq.colour)], &sq.gid, sizeof(sq.gid));
    return true;
  }

  bool insert(poplar::Input<poplar::Vector<float>> &buffer, unsigned idx, struct square& sq) {
    if (idx + sizeof(square) > buffer.size()) {
      return false;
    }
    memcpy((void *) &buffer[idx], &sq, sizeof(sq.centre));
    memcpy((void *) &buffer[idx+sizeof(sq.centre)], &sq.colour, sizeof(sq.colour));
    memcpy((void *) &buffer[idx+sizeof(sq.centre)+sizeof(sq.colour)], &sq.gid, sizeof(sq.gid));
    return true;
  }

  square unpackGaussian(poplar::Input<poplar::Vector<float>> &buffer, unsigned idx) {
    struct square sq;

    ivec4 centre;
    memcpy(&centre, &buffer[idx], sizeof(centre));
    ivec4 colour;
    memcpy(&colour, &buffer[idx+sizeof(centre)], sizeof(colour));
    unsigned gid;
    memcpy(&gid, &buffer[idx+sizeof(centre)+sizeof(colour)], sizeof(gid));

    sq.centre = centre;
    sq.colour = colour;
    sq.gid = gid;
    
    return sq;
  }

  square unpackGaussian(poplar::Output<poplar::Vector<float>> &buffer, unsigned idx) {
    struct square sq;

    ivec4 centre;
    memcpy(&centre, &buffer[idx], sizeof(centre));
    ivec4 colour;
    memcpy(&colour, &buffer[idx+sizeof(centre)], sizeof(colour));
    unsigned gid;
    memcpy(&gid, &buffer[idx+sizeof(centre)+sizeof(colour)], sizeof(gid));

    sq.centre = centre;
    sq.colour = colour;
    sq.gid = gid;
    
    return sq;
  }

    // invalidate a gaussian in the buffer
  void evict(poplar::Input<poplar::Vector<float>> &buffer, unsigned idx) {
    struct square sq;
    sq.gid = 0;
    insert(buffer, idx, sq);
  }

  bool hasCopyIn(struct square& sq, poplar::Input<poplar::Vector<float>> &buffer) {
    for (auto i = 0; i + sizeof(square) < buffer.size(); i+=sizeof(square)) {
      struct square sq2 = unpackGaussian(buffer, i);
      if (sq2.gid == sq.gid) {
        return true;
      }
    }
    return false;
  }

  void send(struct square &sq, directions dirs) {

     auto hasCopy = [this](struct square& sq, poplar::Output<poplar::Vector<float>> &buffer) {
      for (auto i = 0; i < buffer.size(); i+=sizeof(square)) {
        struct square sq2 = unpackGaussian(buffer, i);
        if (sq2.gid == sq.gid) {
          return true;
        }
      }
      return false;
    };

    if (dirs.right && squaresSentRight < rightOut.size() && !hasCopy(sq, rightOut)) {
      insert(rightOut, squaresSentRight, sq);
      squaresSentRight+=sizeof(square);
    }

    if (dirs.left && squaresSentLeft < leftOut.size() && !hasCopy(sq, leftOut)) {
      insert(leftOut, squaresSentLeft, sq);
      squaresSentLeft+=sizeof(square);
    }

    if (dirs.up && squaresSentUp < upOut.size() && !hasCopy(sq, upOut)) {
      insert(upOut, squaresSentUp, sq);
      squaresSentUp+=sizeof(square);
    }

    if (dirs.down && squaresSentDown < downOut.size() && !hasCopy(sq, downOut)) {
      insert(downOut, squaresSentDown, sq);
      squaresSentDown+=sizeof(square);
    }

  }

  bool storeOnTile(struct square &sq) {

    for (auto i = 0; i < squares.size(); i+=sizeof(square)) {
      struct square sq2 = unpackGaussian(squares, i);
      if (sq2.gid == 0u) {
        return insert(squares, i, sq);
      }
    }

    // check i + sizeof(square) < vertsIn.size() because verts in is not multiple of square size
    for (auto i = 0; i + sizeof(square) < vertsIn.size(); i+=sizeof(square)) {
      struct square sq2 = unpackGaussian(vertsIn, i);
      if (sq2.gid == 0u) {
        return insert(vertsIn, i, sq);
      }
    }


    return false;
  }

  void clearFb() {
    for (auto i = 0; i < localFb.size(); i+=4) {
      auto black = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
      memcpy(&localFb[i], glm::value_ptr(black), sizeof(black));
    }
    squaresSentRight = 0;
    squaresSentLeft = 0;
    squaresSentUp = 0;
    squaresSentDown = 0;
  }

  void initGaussians(ivec4& colour, const glm::mat4& m, const std::pair<ivec2, ivec2> tb, const splat::Viewport& vp) {
    auto [tlBound, brBound] = tb;

    // loop over the points originally stored on this tile and initialise the gaussians
    for (auto i = 0; i < vertsIn.size(); i+=4) {
      auto upt = glm::make_vec4(&vertsIn[i]);

      // give point a square extent
      glm::vec2 centre2D = vp.clipSpaceToViewport(m * upt);
      ivec2 topleft = {centre2D.x - (EXTENT / 2.0f), centre2D.y - (EXTENT / 2.0f)};
      ivec2 bottomright = {centre2D.x + (EXTENT / 2.0f), centre2D.y + (EXTENT / 2.0f)};

      // clip the square to the tile, return true 
      // if it needs to be copied to a different direction
      auto dirs = square::clip(tlBound, brBound, topleft, bottomright);

      struct square sq;
      sq.centre = {upt.x, upt.y, upt.z, upt.w};
      sq.gid = (i/4)+1+tile_id[0]*vertsIn.size(); // give unique gaussian id
      sq.colour = colour;
      send(sq, dirs);

      if (dirs.keep) {
        
        for (auto s = 0u; s < squares.size(); s+=sizeof(square)) {
          struct square sq2 = unpackGaussian(squares, s);
          if (sq2.gid == 0u) {
            insert(squares, s, sq);
            break;
          }
        }

        // check i + sizeof(square) < vertsIn.size() because verts in is not multiple of square size
        // for (auto t = 0u; t < i && t + sizeof(square) < vertsIn.size(); t+=sizeof(square)) {
        //   struct square sq2 = unpackGaussian(vertsIn, t);
        //   if (sq2.gid == 0u) {
        //     insert(vertsIn, t, sq);
        //   }
        // }
      }

      gaussiansInitialised++;
     }

  }

  void readBuffer(poplar::Input<poplar::Vector<float>> &bufferIn, const glm::mat4& m, const std::pair<ivec2, ivec2> tb, const splat::Viewport& vp) {
    auto [tlBound, brBound] = tb;

    for (auto i = 0; i < bufferIn.size(); i+=sizeof(square)) {
      struct square sq = unpackGaussian(bufferIn, i);
      if (sq.gid == 0u) {
        break;
      }

      auto upt = glm::vec4(sq.centre.x, sq.centre.y, sq.centre.z, sq.centre.w);
      glm::vec2 centre2D = vp.clipSpaceToViewport(m * upt);
      // give point a square extent
      ivec2 topleft = {centre2D.x - (EXTENT / 2.0f), centre2D.y - (EXTENT / 2.0f)};
      ivec2 bottomright = {centre2D.x + (EXTENT / 2.0f), centre2D.y + (EXTENT / 2.0f)};

      // clip the square to the tile, return true 
      // if it needs to be copied to a different direction
      auto dirs = square::clip(tlBound, brBound, topleft, bottomright);

      // set the topleft and bottomright to be relative to the tile
      viewspaceToTile(topleft, tlBound);
      viewspaceToTile(bottomright, tlBound);


      if (dirs.keep && !hasCopyIn(sq, squares)) { 
        for (auto s = 0u; s < squares.size(); s+=sizeof(square)) {
          struct square sq2 = unpackGaussian(squares, s);
          if (sq2.gid == 0u) {
            insert(squares, s, sq);
            break;
          }
        }
      } 
      send(sq, dirs);
    }
  }

  void renderStored(poplar::Input<poplar::Vector<float>> &bufferIn, const glm::mat4& m, const std::pair<ivec2, ivec2> tb, const splat::Viewport& vp) {
    auto [tlBound, brBound] = tb;

    for (auto i = 0; i < bufferIn.size(); i+=sizeof(square)) {
      struct square sq = unpackGaussian(bufferIn, i);
      if (sq.gid == 0u) {
        continue;
      }

      auto upt = glm::vec4(sq.centre.x, sq.centre.y, sq.centre.z, sq.centre.w);
      glm::vec2 centre2D = vp.clipSpaceToViewport(m * upt);
      // give point a square extent
      ivec2 topleft = {centre2D.x - (EXTENT / 2.0f), centre2D.y - (EXTENT / 2.0f)};
      ivec2 bottomright = {centre2D.x + (EXTENT / 2.0f), centre2D.y + (EXTENT / 2.0f)};

      // clip the square to the tile, return true 
      // if it needs to be copied to a different direction
      auto dirs = square::clip(tlBound, brBound, topleft, bottomright);

      // set the topleft and bottomright to be relative to the tile
      viewspaceToTile(topleft, tlBound);
      viewspaceToTile(bottomright, tlBound);

      if (dirs.keep) {
        splat(sq.colour, topleft, bottomright);
      } else {
        evict(bufferIn, i);
      }
      send(sq, dirs);

    }
  }

  bool compute(unsigned workerId) {
    if (workerId != 0) {
      return true;
    }

    // construct mapping from tile to framebuffer
    const TiledFramebuffer fbMapping(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    // Transpose because GLM storage order is column major:
    const auto m = glm::transpose(glm::make_mat4(&matrix[0]));
    const auto tb = fbMapping.getTileBounds(tile_id[0]);
    // zero the framebuffer and clear the send buffers
    clearFb();

    ivec4 tidColour = {0.0f, 1.0f, tile_id[0] * (1.0f / fbMapping.numTiles), 0.0f};


    if (gaussiansInitialised < vertsIn.size() / sizeof(square)) {
      // initialise the gaussians from the pts
      // sends gaussians to other tiles, or stores in local memory
      // recover the original vector for extra gaussian storage
      initGaussians(tidColour, m, tb, vp);
    } 

    // render anything inside the local tile memory
    // renderStored(vertsIn, m, tb, vp);

    // read the gaussians from the send buffers,
    // project and clip, 
    // send to other tiles if need 
    renderStored(squares, m, tb, vp);
    readBuffer(rightIn, m, tb, vp);
    readBuffer(leftIn, m, tb, vp);
    readBuffer(upIn, m, tb, vp);
    readBuffer(downIn, m, tb, vp);

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
