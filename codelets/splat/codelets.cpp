// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/StackSizeDefs.hpp>

#include <glm/mat4x4.hpp>
#include <glm/gtc/type_ptr.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/tileMapping/tile_config.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/viewport.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/splat/ipu_geometry.hpp>

using namespace splat;

#ifdef __IPU__
#include <ipu_vector_math>
#include <ipu_memory_intrinsics>
#include <ipu_builtins.h>
#endif

DEF_FUNC_CALL_PTRS("_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE", "_ZNK5splat10Gaussian2D14getBoundingBoxEv,_ZNK5splat10Gaussian2D6insideEff");

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
  poplar::InOut<poplar::Vector<float>> vertsIn;

  poplar::Output<poplar::Vector<float>> localFb;
  poplar::Input<poplar::Vector<int>> tile_id;

  poplar::Input<poplar::Vector<float>> rightIn;
  poplar::Output<poplar::Vector<float>> rightOut;

  poplar::Input<poplar::Vector<float>> leftIn; 
  poplar::Output<poplar::Vector<float>> leftOut;

  poplar::Input<poplar::Vector<float>> upIn;
  poplar::Output<poplar::Vector<float>> upOut;

  poplar::Input<poplar::Vector<float>> downIn;
  poplar::Output<poplar::Vector<float>> downOut;

  poplar::InOut<poplar::Vector<float>> stored;

  bool init;

  unsigned toByteBufferIndex(float x, float y) {
    return unsigned(x + y * IPU_TILEWIDTH) * 4;
  } 

  void setPixel(float x, float y, const ivec4 &colour) {
    ivec4 pixel;
    unsigned idx = toByteBufferIndex(x, y);
    if (idx >= localFb.size()) {
      printf("ERROR: setting pixel outside of framebuffer bounds\n");
      printf(" --> please check coordinate mapping is in tile space\n");
      return;
    }
    memcpy(&pixel, &localFb[idx], sizeof(pixel));
    pixel = pixel + colour;
    memcpy(&localFb[idx], &pixel, sizeof(pixel));
  }

  ivec2 viewspaceToTile(const ivec2& pt, ivec2 tlBound) {
    return {floor(pt.x - tlBound.x), floor(pt.y - tlBound.y)};
  }

  ivec4 getTileColour() {
    ivec4 c;
    if (tile_id[0] % 3 == 0) {
      c = {.3f, 0.0f, 0.0f, 0.0f};
    } else if (tile_id[0] % 3 == 1) {
      c = {0.0f, .3f, 0.0f, 0.0f};
    } else {
      c = {0.0f, 0.0f, .3f, 0.0f};
    }
    return c;
  }

  // bool tileContainsBoundary(Primitive &p, const Bounds2f& tb) {
  //   return (p.inside(tb.min.x, tb.min.y) || p.inside(tb.min.x, tb.max.y) || p.inside(tb.max.x, tb.min.y) || p.inside(tb.max.x, tb.max.y));
  // }

  bool insertAt(poplar::Vector<float> &buffer, unsigned idx, Gaussian3D& g) {
    if (idx + 16 > buffer.size()) {
      return false;
    }
    memcpy(&buffer[idx], &g.mean, sizeof(g.mean));
    memcpy(&buffer[idx+4], &g.colour, sizeof(g.colour));
    memcpy(&buffer[idx+8], &g.gid, sizeof(g.gid));
    memcpy(&buffer[idx+9], &g.scale, sizeof(g.scale));
    memcpy(&buffer[idx+12], &g.rot, sizeof(g.rot));
    return true;
  }

  bool insertAt(poplar::InOut<poplar::Vector<float>> &buffer, unsigned idx, Gaussian3D& g) {
    if (idx + 16 > buffer.size()) {
      return false;
    }
    memcpy((void *) &buffer[idx], &g.mean, sizeof(g.mean));
    memcpy((void *) &buffer[idx+4], &g.colour, sizeof(g.colour));
    memcpy((void *) &buffer[idx+8], &g.gid, sizeof(g.gid));
    memcpy((void *) &buffer[idx+9], &g.scale, sizeof(g.scale));
    memcpy((void *) &buffer[idx+12], &g.rot, sizeof(g.rot));
    return true;
  }

  bool insert(poplar::InOut<poplar::Vector<float>> &buffer, Gaussian3D& g) {
    unsigned idx = buffer.size();
    for (auto i = 0; i < buffer.size(); i+=16) {
      float gid;
      memcpy(&gid, &buffer[i+8], sizeof(gid)); // TODO: specify gid
      if (gid == g.gid) {
        return false;
      }
      if (gid == 0u && i < idx) {
        idx = i;
      }
    }
    return insertAt(buffer, idx, g);
  }

  bool insert(poplar::Vector<float> &buffer, Gaussian3D& g) {
    unsigned idx = buffer.size();
    for (auto i = 0; i < buffer.size(); i+=16) {
      float gid;
      memcpy(&gid, &buffer[i+8], sizeof(gid)); // TODO: specify gid
      if (gid == g.gid) {
        return insertAt(buffer, i, g);
      }
      if (gid == 0u && i < idx) {
        idx = i;
      }
    }
    return insertAt(buffer, idx, g);
  }

  // TODO: change to use templates instead of inheritance as virtual function insert 
  // vtable pointer so unpacking structs needs to be shifted by 1
  Gaussian3D unpackGaussian3D(poplar::InOut<poplar::Vector<float>> &buffer, unsigned idx, const glm::mat4& viewmatrix, const splat::Viewport& vp) {

    ivec4 mean;
    memcpy(&mean, &buffer[idx], sizeof(mean));
    ivec4 colour;
    memcpy(&colour, &buffer[idx+4], sizeof(colour));
    float gid;
    memcpy(&gid, &buffer[idx+8], sizeof(gid));
    ivec3 scale;
    memcpy(&scale, &buffer[idx+9], sizeof(scale));
    ivec4 rot;
    memcpy(&rot, &buffer[idx+12], sizeof(rot));

    struct Gaussian3D g;
    g.mean = mean;
    g.scale = scale; 
    g.rot = rot;
    g.colour = colour;
    g.gid = gid;

    return g;
  }

  Gaussian3D unpackGaussian3D(poplar::Input<poplar::Vector<float>> &buffer, unsigned idx, const glm::mat4& viewmatrix, const splat::Viewport& vp) {

    ivec4 mean;
    memcpy(&mean, &buffer[idx], sizeof(mean));
    ivec4 colour;
    memcpy(&colour, &buffer[idx+4], sizeof(colour));
    float gid;
    memcpy(&gid, &buffer[idx+8], sizeof(gid));
    ivec3 scale;
    memcpy(&scale, &buffer[idx+9], sizeof(scale));
    ivec4 rot;
    memcpy(&rot, &buffer[idx+12], sizeof(rot));

    struct Gaussian3D g;
    g.mean = mean;
    g.scale = scale; 
    g.rot = rot;
    g.colour = colour;
    g.gid = gid;

    return g;
  }

    // invalidate a gaussian in the buffer
  void evict(poplar::Vector<float> &buffer, unsigned idx) {
    Gaussian3D g;
    g.gid = 0.f;
    insertAt(buffer, idx, g);
  }

  void evict(poplar::InOut<poplar::Vector<float>> &buffer, unsigned idx) {
    Gaussian3D g;
    g.gid = 0.f;
    insertAt(buffer, idx, g);
  }

  void send(Gaussian3D &g, directions dirs) {
    if (dirs.right) {
      insert(rightOut, g);
    }
    if (dirs.left) {
      insert(leftOut, g);
    }
    if (dirs.up) {
      insert(upOut, g);
    }
    if (dirs.down) {
      insert(downOut, g);
    }
  }

  void sendOnce(Gaussian3D &g, directions possibleDirs, direction recievedDirection) {
    if (recievedDirection != direction::right && possibleDirs.right) {
      insert(rightOut, g);
    } else if (recievedDirection != direction::left && possibleDirs.left) {
      insert(leftOut, g);
    } else if (recievedDirection != direction::up && possibleDirs.up) {
      insert(upOut, g);
    } else if (recievedDirection != direction::down && possibleDirs.down) {
      insert(downOut, g);
    } 
  }

  void colourFb(const ivec4 &colour, unsigned workerId) {
    const auto startIndex = 4 * workerId;
    for (auto i = startIndex; i < localFb.size(); i += 4 * numWorkers()) {
      memcpy(&localFb[i], &colour, sizeof(colour));
    }
  }

  void colourFb(const ivec4 &colour) {
    for (auto i = 0; i < localFb.size(); i += 4) {
      memcpy(&localFb[i], &colour, sizeof(colour));
    }
  }

  unsigned rasterise(const Gaussian2D &g, const Bounds2f& bb, const Bounds2f& tb) {
    auto tc = getTileColour();
    auto count = 0u;
    for (auto i = bb.min.x; i < bb.max.x; i++) {
      for (auto j = bb.min.y; j < bb.max.y; j++) {
        auto px = viewspaceToTile({i, j}, tb.min);
        if(g.inside(i,j)) {
          setPixel(px.x, px.y, g.colour);
        } else {
          // setPixel(px.x, px.y, tc);
        }
        count++;
      }
    }
    return count;
  }

  void renderMain(poplar::InOut<poplar::Vector<float>> &bufferIn, const glm::mat4& viewmatrix, const TiledFramebuffer& tfb, const splat::Viewport& vp) {
    const auto tb = tfb.getTileBounds(tile_id[0]);
    for (auto i = 0; i < bufferIn.size(); i+=16) {
      Gaussian3D g = unpackGaussian3D(bufferIn, i, viewmatrix, vp);
      if (g.gid <= 0) {
        break;
      }

      ivec3 cov2D = g.ComputeCov2D(viewmatrix, 1.0f, 1.0f);
      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, g.mean.w};
      auto projMean = vp.clipSpaceToViewport(viewmatrix * glmMean);
      Gaussian2D g2D({projMean.x, projMean.y}, g.colour, cov2D);

      auto bb = g2D.GetBoundingBox();

      directions dirs;
      bb = bb.clip(tb, dirs);

      if (tb.contains(g2D.mean)) { // is anchored 
        rasterise(g2D, bb, tb);
        send(g, dirs);
      } else {
        // send in any direction once
        sendOnce(g, dirs, direction::none);
        evict(bufferIn, i);
      }
    }
  }

  void readInput(poplar::Input<poplar::Vector<float>> &bufferIn,
                                    const direction& recievedFrom,
                                    const glm::mat4& viewmatrix,
                                    const TiledFramebuffer& tfb,
                                    const splat::Viewport& vp) {

    const auto tb = tfb.getTileBounds(tile_id[0]);
    const auto fromTile = tfb.getNearbyTile(tile_id[0], recievedFrom);
    const auto fromTb = tfb.getTileBounds(fromTile);

    for (auto i = 0; i < bufferIn.size(); i+=16) {
      Gaussian3D g = unpackGaussian3D(bufferIn, i, viewmatrix, vp);
      if (g.gid == 0) {
        break;
      }

      auto green = ivec4{0.0f, 0.3f, 0.0f, 0.0f};
      colourFb(green);

      ivec3 cov2D = g.ComputeCov2D(viewmatrix, 1.0f, 1.0f);
      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, g.mean.w};
      auto projMean = vp.clipSpaceToViewport(viewmatrix * glmMean);
      Gaussian2D g2D({projMean.x, projMean.y}, g.colour, cov2D);

      // see whether the gaussian is closer to its destination anchor
      unsigned dest = tfb.pixCoordToTile(projMean.x, projMean.y);
      const auto destTb = tfb.getTileBounds(dest);
      const auto curDist = ivec2::manhattanDistance(destTb.min, tb.min); // min or centroid 
      const auto prevDist = ivec2::manhattanDistance(destTb.min, fromTb.min); // min or centroid
      bool closer = curDist < prevDist;

      auto bb = g2D.GetBoundingBox();
      directions dirs;
      bb = bb.clip(tb, dirs);

      if (tb.contains(g2D.mean)) { 
        // is anchored then we will render and store 
        insert(vertsIn, g);
      } else if (bb.diagonal().length() < 1) {
        // if not anchored and the visible region is 0
        // then we send in a direction that we didn't recieved it from
        sendOnce(g, dirs, recievedFrom);
      } else {
        rasterise(g2D, bb, tb);
        insert(stored, g);
      }
    }
  } 

  void renderStored(poplar::InOut<poplar::Vector<float>> &bufferIn, const glm::mat4& viewmatrix, const TiledFramebuffer& tfb, const splat::Viewport& vp) {
    const auto tb = tfb.getTileBounds(tile_id[0]);
    for (auto i = 0; i < bufferIn.size(); i+=16) {
      Gaussian3D g = unpackGaussian3D(bufferIn, i, viewmatrix, vp);
      if (g.gid <= 0) {
        break;
      }

      ivec3 cov2D = g.ComputeCov2D(viewmatrix, 1.0f, 1.0f);
      glm::vec4 glmMean = {g.mean.x, g.mean.y, g.mean.z, g.mean.w};
      auto projMean = vp.clipSpaceToViewport(viewmatrix * glmMean);
      Gaussian2D g2D({projMean.x, projMean.y}, g.colour, cov2D);

      auto bb = g2D.GetBoundingBox();

      directions dirs;
      bb = bb.clip(tb, dirs);

      if (tb.contains(g2D.mean)) { // is anchored 
        insert(vertsIn, g);
      } else if (bb.diagonal().length() < 1) {
        evict(bufferIn, i);
      } else {
        // rasterise(g2D, bb, tb);
      }
    }
  }

  bool compute(unsigned workerId) {

    // zero the framebuffer and clear the send buffers
    colourFb({0.0f, 0.0f, 0.0f, 0.0f}, workerId);

    //clear all of the out buffers:
    for (auto i = 0; i < rightOut.size(); i+=16) {
      evict(rightOut, i);
    }
    for (auto i = 0; i < leftOut.size(); i+=16) {
      evict(leftOut, i);
    }
    for (auto i = 0; i < upOut.size(); i+=16) {
      evict(upOut, i);
    }
    for (auto i = 0; i < downOut.size(); i+=16) {
      evict(downOut, i);
    }
  
    if (workerId != 0) {
      return true;
    }

    // construct mapping from tile to framebuffer
    const TiledFramebuffer tfb(IPU_TILEWIDTH, IPU_TILEHEIGHT);
    const splat::Viewport vp(0.0f, 0.0f, IMWIDTH, IMHEIGHT);
    // Transpose because GLM storage order is column major:
    const auto viewmatrix = glm::transpose(glm::make_mat4(&matrix[0]));


    readInput(rightIn, direction::right, viewmatrix, tfb, vp);
    readInput(leftIn, direction::left, viewmatrix, tfb, vp);
    readInput(upIn, direction::up, viewmatrix, tfb, vp);
    readInput(downIn, direction::down, viewmatrix, tfb, vp);

    renderMain(vertsIn, viewmatrix, tfb, vp);
    renderStored(stored, viewmatrix, tfb, vp);

   
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
