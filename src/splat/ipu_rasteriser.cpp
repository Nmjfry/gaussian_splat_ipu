// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/ipu_rasteriser.hpp>
#include <splat/geometry.hpp>
#include <ipu/io_utils.hpp>

#include <glm/gtc/type_ptr.hpp>

#include <poputil/TileMapping.hpp>

using namespace poplar;

namespace splat {

/// Name the streamable tensors and take a reference to the point data:
IpuSplatter::IpuSplatter(const Points& verts)
  : modelViewProjection("mvp"), inputVertices("verts_in"), outputVertices("verts_out"),
    transformMatrix(16),
    initialised(false)
{
  hostVertices.reserve(4 * verts.size());
  for (const auto& v : verts) {
    hostVertices.push_back(v.p.x);
    hostVertices.push_back(v.p.y);
    hostVertices.push_back(v.p.z);
    hostVertices.push_back(1.f);
  }
}

void IpuSplatter::updateModelViewProjection(const glm::mat4& mvp) {
  auto mvpt = glm::transpose(mvp);
  auto ptr = (const float*)glm::value_ptr(mvpt);
  for (auto i = 0u; i < transformMatrix.size(); ++i) {
    transformMatrix[i] = *ptr;
    ptr += 1;
  }
}

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

/// Distribute elements across tiles and add padding such that the number of elements on a
/// tile is always divisible by grainSize (padding is added to the last tile guarantee this).
///
/// Returns a padded and correctly tile mapped tensor.
poplar::Tensor applyTileMappingAndPad(poplar::Graph& g, const poplar::Tensor& input, std::size_t grainSize) {
  ipu_utils::logger()->info("Input num elements: {}", input.numElements());
  const double numTiles = g.getTarget().getNumTiles();
  double grainsPerTile = std::ceil(input.numElements() / (numTiles * grainSize));
  double elementsPerTile = grainsPerTile * grainSize;
  double fullTiles = std::floor(input.numElements() / elementsPerTile);
  double remainingElements = input.numElements() - (fullTiles * elementsPerTile);
  double paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;

  ipu_utils::logger()->info("Upper bound elements per tile: {}", elementsPerTile);
  ipu_utils::logger()->info("Full tiles: {}", fullTiles);
  ipu_utils::logger()->info("Remaining elements: {}", remainingElements);
  ipu_utils::logger()->info("Padded elements on last tile: {}", paddedRemainder);
  ipu_utils::logger()->info("Padding: {}", paddedRemainder - remainingElements);

  // Add zero padding:
  const std::size_t padding = paddedRemainder - remainingElements;
  auto zeroPadding = g.addConstant(input.elementType(), {padding}, 0, "vert_padding");

  auto paddedInput = poplar::concat({input, zeroPadding}, 0);
  ipu_utils::logger()->info("Padded shape: {}", paddedInput.shape());

  auto sliceStart = 0u;
  auto t = 0u;
  std::size_t totalTiles = fullTiles;
  for (t; t < totalTiles; ++t) {
    const auto sliceEnd = sliceStart + elementsPerTile;
    g.setTileMapping(paddedInput.slice(sliceStart, sliceEnd), t);
    sliceStart = sliceEnd;
  }

  // Last tile has fewer elements:
  g.setTileMapping(paddedInput.slice(sliceStart, sliceStart + paddedRemainder), t);

  return paddedInput;
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

  // Map the point cloud vertices across all tiles but specify a grain size so that 4 vectors
  // do not get split in the middle:
  inputVertices.buildTensor(vg, FLOAT, {hostVertices.size()});
  auto paddedInput = applyTileMappingAndPad(vg, inputVertices.get(), 4 * 8);

  // Clone the input to make the output:
  auto paddedOutput = vg.clone(paddedInput);
  outputVertices = paddedOutput.slice(0u, inputVertices.numElements());

  // Build a compute set to transform the points:
  auto projectCs = vg.addComputeSet("project");

  // Get the tile mapping and connect the vertices:
  const auto tm = vg.getTileMapping(paddedInput);
  for (auto t = 0u; t < tm.size(); ++t) {
    const auto& m = tm[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected vertices to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      // Add a transform vertex to process the points connecting the same slices of input and output:
      auto v = vg.addVertex(projectCs, "Transform4x4_intrinsics");
      vg.setTileMapping(v, t);
      auto sliceIn = paddedInput.slice(m.front());
      auto sliceOut = paddedOutput.slice(m.front());
      vg.connect(v["vertsIn"], sliceIn);
      vg.connect(v["vertsOut"], sliceOut);

      const auto vertsThisTile = sliceIn.numElements() / 4;
      if (vertsThisTile % 8 != 0) {
        ipu_utils::logger()->error("Tile {} has {} vertices which is not a multiple of 8", t, vertsThisTile);
        throw std::runtime_error("Vertices per tile must be a multiple of 8 to use the AMP.");
      }

      // Add the tile local MVP matrix variable and add to the broadcast program:
      auto localMvp = vg.clone(modelViewProjection, "mvp_tile_" + std::to_string(t));
      vg.setTileMapping(localMvp, t);
      vg.connect(v["matrix"], localMvp.flatten());
      broadcastMvp.add(program::Copy(modelViewProjection, localMvp));
    }
  }

  program::Sequence main;
  main.add(broadcastMvp);
  main.add(program::Execute(projectCs));
  main.add(outputVertices.buildRead(vg, true));

  getPrograms().add("write_verts", inputVertices.buildWrite(vg, true));
  getPrograms().add("project", main);
}

void IpuSplatter::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!initialised) {
    initialised = true;
    modelViewProjection.connectWriteStream(engine, transformMatrix);
    inputVertices.connectWriteStream(engine, hostVertices);
    outputVertices.connectReadStream(engine, hostVertices);
    getPrograms().run(engine, "write_verts");
  }

  getPrograms().run(engine, "project");
}

} // end of namespace splat
