// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/ipu_rasteriser.hpp>
#include <splat/geometry.hpp>
#include <ipu/io_utils.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <poputil/TileMapping.hpp>

using namespace poplar;

namespace splat {

/// Name the streamable tensors and take a reference to the point data:
IpuSplatter::IpuSplatter(const Points& verts, bool noAMP)
  : modelViewProjection("mvp"), inputVertices("verts_in"), outputVertices("verts_out"),
    transformMatrix(16),
    initialised(false),
    disableAMPVertices(noAMP)
{
  hostVertices.reserve(4 * verts.size());
  for (const auto& v : verts) {
    hostVertices.push_back(v.p.x);
    hostVertices.push_back(v.p.y);
    hostVertices.push_back(v.p.z);
    hostVertices.push_back(1.f);
  }
}

// IpuSplatter::IpuSplatter(const Points& verts, splat::TiledFramebuffer& fb, bool noAMP)
//   : modelViewProjection("mvp"), inputVertices("verts_in"), outputVertices("verts_out"),
//     transformMatrix(16),
//     initialised(false),
//     disableAMPVertices(noAMP)
// {
//   hostVertices.reserve(4 * verts.size());
//   for (const auto& v : verts) {
//     hostVertices.push_back(v.p.x);
//     hostVertices.push_back(v.p.y);
//     hostVertices.push_back(v.p.z);
//     hostVertices.push_back(1.f);
//   }
//   frameBuffer.resize(fb.width * fb.height * 4, 0.0f);
// }

void IpuSplatter::updateModelViewProjection(const glm::mat4& mvp) {
  auto mvpt = glm::transpose(mvp);
  auto ptr = (const float*)glm::value_ptr(mvpt);
  for (auto i = 0u; i < transformMatrix.size(); ++i) {
    transformMatrix[i] = *ptr;
    ptr += 1;
  }
}


// void IpuSplatter::getFramebuffer(cv::Mat& frame) const {
// need to ensure that we read sections of the framebuffer 
// as square tiles and then stitch them together

//  unsigned imWidth = frame.cols;
//  unsigned imHeight = frame.rows;

// 

//   const auto* ptr = frameBuffer.data();
//   #pragma omp parallel for schedule(static, 128) num_threads(24)
//   for (auto i = 0u; i < frameBuffer.size(); ++i, ptr += 4) {
//     glm::vec4 pixel; 

//     pixel.b = *(ptr + 0);
//     pixel.g = *(ptr + 1);
//     pixel.r = *(ptr + 2);
//     pixel.a = 1.0;

//     const auto colour = cv::Vec3b((uint8_t) pixel.b, (uint8_t) pixel.g, (uint8_t) pixel.r);
//     // Convert from pixel vector to pixel coords:
//     std::uint32_t r = i / frame.cols;
//     std::uint32_t c = i - r * frame.cols;

//     // Clip points to the frame and splat:
//     if (r < frame.rows && c < frame.cols) {
//       frame.at<cv::Vec3b>(r, c) = colour;

//       // #pragma omp atomic update
//     }
//   }

// }

void IpuSplatter::getProjectedPoints(std::vector<glm::vec4>& pts) const {
  pts.resize(hostVertices.size() / 4);
  const auto* ptr = hostVertices.data();
  for (auto i = 0u; i < pts.size(); ++i) {
    pts[i].x = *(ptr + 0);
    pts[i].y = *(ptr + 1);
    pts[i].z = *(ptr + 2);
    pts[i].w = *(ptr + 3);
    ptr += 4;
  }
}

struct MappingInfo {
  std::size_t padding;
  std::size_t elementsPerTile;
  std::size_t totalTiles;
};

MappingInfo calculateMapping(poplar::Graph& g, std::size_t numElements, std::size_t grainSize) {
  ipu_utils::logger()->info("Input num elements: {}", numElements);
  const double numTiles = g.getTarget().getNumTiles();
  double grainsPerTile = std::ceil(numElements / (numTiles * grainSize));
  double elementsPerTile = grainsPerTile * grainSize;
  double fullTiles = std::floor(numElements / elementsPerTile);
  double remainingElements = numElements - (fullTiles * elementsPerTile);
  double paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;

  ipu_utils::logger()->info("Upper bound elements per tile: {}", elementsPerTile);
  ipu_utils::logger()->info("Full tiles: {}", fullTiles);
  ipu_utils::logger()->info("Remaining elements: {}", remainingElements);
  ipu_utils::logger()->info("Padded elements on last tile: {}", paddedRemainder);
  ipu_utils::logger()->info("Padding: {}", paddedRemainder - remainingElements);

  const std::size_t padding = paddedRemainder - remainingElements;
  return MappingInfo{padding, std::size_t(elementsPerTile), std::size_t(fullTiles)};
}

/// Distribute elements across tiles such that the number of elements on a
/// tile is always divisible by grainSize. The tensor must have already been padded
/// to a multiple of grain size.
void applyTileMapping(poplar::Graph& g, const poplar::Tensor& paddedInput, const MappingInfo& info) {
  auto sliceStart = 0u;
  auto t = 0u;
  for (t; t < info.totalTiles; ++t) {
    const auto sliceEnd = sliceStart + info.elementsPerTile;
    g.setTileMapping(paddedInput.slice(sliceStart, sliceEnd), t);
    sliceStart = sliceEnd;
  }

  // Last tile has fewer elements:
  auto lastSlice = paddedInput.slice(sliceStart, paddedInput.numElements());
  ipu_utils::logger()->info("Size of slice on last tile: {}", lastSlice.numElements());
  g.setTileMapping(lastSlice, t);
}

// Add a vertex to project vertices that uses vanilla C++ code.
void addProjectionVertex(poplar::Graph& g, poplar::ComputeSet& cs, unsigned t,
                         const poplar::Tensor& modelViewProjection, const poplar::Tensor& sliceIn, const poplar::Tensor& sliceOut) {
  auto v = g.addVertex(cs, "Transform4x4");
  g.setTileMapping(v, t);

  g.connect(v["matrix"], modelViewProjection);
  g.connect(v["vertsIn"], sliceIn);
  g.connect(v["vertsOut"], sliceOut);
}

// Add a vertex to project vertices that uses is optimised using the tile's AMP engine.
void addProjectionVertexAMP(poplar::Graph& g, poplar::ComputeSet& cs, unsigned t,
                        const poplar::Tensor& modelViewProjection, const poplar::Tensor& sliceIn, const poplar::Tensor& sliceOut) {
  auto v = g.addVertex(cs, "Transform4x4_amp");
  auto vs = g.addVertex(cs, "LoadMatrix");
  g.setTileMapping(v, t);
  g.setTileMapping(vs, t);

  g.connect(vs["matrix"], modelViewProjection);
  g.connect(v["vertsIn"], sliceIn);
  g.connect(v["vertsOut"], sliceOut);

  const auto vertsThisTile = sliceIn.numElements() / 4;
  if (vertsThisTile % 8 != 0) {
    ipu_utils::logger()->error("Tile {} has {} vertices which is not a multiple of 8", t, vertsThisTile);
    throw std::runtime_error("Vertices per tile must be a multiple of 8 to use the AMP.");
  }
}

void IpuSplatter::build(poplar::Graph& graph, const poplar::Target& target) {
  auto vg = graph.createVirtualGraph(0u, 1440u);

  const auto codeletFile = std::string(POPC_PREFIX) + "/codelets/splat/codelets.cpp";
  const auto glmPath = std::string(POPC_PREFIX) + "/external/glm/";
  const auto otherIncludes = std::string(POPC_PREFIX) + "/include/missing";
  const auto includes = " -I " + glmPath + " -I " + otherIncludes;
  ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
  vg.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);

  // Create storage for the model view projeciton matrix. Place the master copy on tile 0
  // and then broadcast from their to all other tiles before any computations.
  modelViewProjection.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(modelViewProjection, 0u);

  // Build a program to upload and broadcast the modelling-projection matrix:
  program::Sequence broadcastMvp;
  broadcastMvp.add(modelViewProjection.buildWrite(vg, true));

  // Map the point cloud vertices across all tiles. If we are not using AMP the only constraint
  // is that the grain size must be a multiple of 4 (so that 4-vectors are not split between
  // tiles). If we use the AMP we need to have at least 8 4-vectors to fill the AMP pipeline so
  // the minimum grain size is 32:
  const auto grainSize = disableAMPVertices ? 4 : 4 * 8;
  auto mapping = calculateMapping(vg, hostVertices.size(), grainSize);
  auto paddedInput = vg.addVariable(FLOAT, {hostVertices.size() + mapping.padding}, "padded_verts_in");
  applyTileMapping(vg, paddedInput, mapping);

  // We only want to stream to a slice of the padded tensor:
  inputVertices = paddedInput.slice(0, hostVertices.size());

  ipu_utils::logger()->info("Size input: {}", inputVertices.numElements());
  ipu_utils::logger()->info("Size of padded input: {}", paddedInput.numElements());


  // ###### TEMPORARY ######
  // Clone the input to make the output:
  auto paddedOutput = vg.clone(paddedInput, "verts_out");
  outputVertices = paddedOutput.slice(0u, inputVertices.numElements());
  // ###### TEMPORARY ######

  // ## proposed change ##
  // auto fbGrainSize = 4;
  // auto framebufferMapping = calculateMapping(vg, frameBuffer.size(), fbGrainSize);
  // auto paddedFrameBuffer = vg.addVariable(FLOAT, {frameBuffer.size() + framebufferMapping.padding}, "padded_frame_buffer");
  // applyTileMapping(vg, paddedFrameBuffer, framebufferMapping);
  // auto outputFrameBuffer = paddedFrameBuffer.slice(0u, frameBuffer.size());
  // #####################


  // Build a compute set to transform the points:
  const auto csName = disableAMPVertices ? "project" : "project_amp";
  auto projectCs = vg.addComputeSet(csName);

  // Get the tile mapping and connect the vertices:
  const auto tm = vg.getTileMapping(paddedInput);

  // ## proposed change ##
  // const auto tmFb = vg.getTileMapping(paddedFrameBuffer);
  // #####################

  for (auto t = 0u; t < tm.size(); ++t) {
    ipu_utils::logger()->info("TILE : {} / {}", t, tm.size());
    const auto& m = tm[t];
    // const auto& mFb = tmFb[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected vertices to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      // Add the tile local MVP matrix variable and append a copies that broadcast it to all tiles:
      auto localMvp = vg.clone(modelViewProjection, "mvp_tile_" + std::to_string(t));
      vg.setTileMapping(localMvp, t);
      broadcastMvp.add(program::Copy(modelViewProjection, localMvp));

      auto sliceIn  = paddedInput.slice(m.front());
      auto sliceOut = paddedOutput.slice(m.front());
      // auto sliceFb = paddedFrameBuffer.slice(mFb.front());

      if (disableAMPVertices) {
        ipu_utils::logger()->warn("AMP vertex disabled on tile: {}", t);
        addProjectionVertex(vg, projectCs, t, localMvp.flatten(), sliceIn, sliceOut);
      } else {
        addProjectionVertexAMP(vg, projectCs, t, localMvp.flatten(), sliceIn, sliceOut);
      }
    }
  }

  program::Sequence main;
  main.add(broadcastMvp);
  main.add(program::Execute(projectCs));
  main.add(outputVertices.buildRead(vg, true));
  // main.add(outputFrameBuffer.buildRead(vg, true));

  getPrograms().add("write_verts", inputVertices.buildWrite(vg, true));
  getPrograms().add("project", main);
}

void IpuSplatter::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!initialised) {
    initialised = true;
    modelViewProjection.connectWriteStream(engine, transformMatrix);
    inputVertices.connectWriteStream(engine, hostVertices);
    // outputVertices.connectReadStream(engine, frameBuffer);
    outputVertices.connectReadStream(engine, hostVertices);
    getPrograms().run(engine, "write_verts");
  }

  getPrograms().run(engine, "project");
}

} // end of namespace splat
