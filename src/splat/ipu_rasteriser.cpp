// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <splat/ipu_rasteriser.hpp>
#include <splat/geometry.hpp>
#include <ipu/io_utils.hpp>
#include <opencv2/highgui.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <poputil/TileMapping.hpp>

#include <tileMapping/edge_builder.hpp>

using namespace poplar;

namespace splat {

IpuSplatter::IpuSplatter(const Points& verts, TiledFramebuffer& fb, bool noAMP)
  : modelViewProjection("mvp"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"), 
    transformMatrix(16),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  hostVertices.reserve(4 * verts.size());
  printf("VERTS size: %luB\n", verts.size());
  for (const auto& v : verts) {
    hostVertices.push_back(v.p.x);
    hostVertices.push_back(v.p.y);
    hostVertices.push_back(v.p.z);
    hostVertices.push_back(1.f);
  }
  frameBuffer.reserve(fb.width * fb.height * 4);
  for (uint i = 0; i < fb.width * fb.height; ++i) {
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
  }
  printf("Fb size: %luB\n", frameBuffer.size());
}

IpuSplatter::IpuSplatter(const Gaussians& verts, TiledFramebuffer& fb, bool noAMP)
  : modelViewProjection("mvp"), inputVertices("verts_in"), outputFramebuffer("frame_buffer"), 
    transformMatrix(16),
    initialised(false),
    disableAMPVertices(noAMP),
    fbMapping(fb)
{
  hostVertices.reserve(GAUSSIAN_SIZE * verts.size());
  printf("num verts in: %lu, elemsize: %lu \n", verts.size(), GAUSSIAN_SIZE);
  for (const auto& v : verts) {
    hostVertices.push_back(v.mean.x);
    hostVertices.push_back(v.mean.y);
    hostVertices.push_back(v.mean.z);
    hostVertices.push_back(v.mean.w);
    hostVertices.push_back(v.colour.x);
    hostVertices.push_back(v.colour.y);
    hostVertices.push_back(v.colour.z);
    hostVertices.push_back(v.colour.w);
    hostVertices.push_back(v.gid);
    hostVertices.push_back(v.scale.x);
    hostVertices.push_back(v.scale.y);
    hostVertices.push_back(v.scale.z);
    hostVertices.push_back(v.rot.x);
    hostVertices.push_back(v.rot.y);
    hostVertices.push_back(v.rot.z);
    hostVertices.push_back(v.rot.w);
  }
  frameBuffer.reserve(fb.width * fb.height * GAUSSIAN_SIZE);
  for (uint i = 0; i < fb.width * fb.height; ++i) {
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
    frameBuffer.push_back(0.0);
  }
  printf("Fb size: %luB\n", frameBuffer.size());
}

void IpuSplatter::updateModelViewProjection(const glm::mat4& mvp) {
  auto mvpt = glm::transpose(mvp);
  auto ptr = (const float*)glm::value_ptr(mvpt);
  for (auto i = 0u; i < transformMatrix.size(); ++i) {
    transformMatrix[i] = *ptr;
    ptr += 1;
  }
}

void IpuSplatter::updateGaussianParams(const Gaussian3D& g) {
  for (auto i = 0u; i < hostVertices.size(); i+=16) {
    // ivec4 mean; // in world space
    // ivec4 colour; // RGBA colour space
    // unsigned gid;
    // ivec2 scale;
    // ivec4 rot;  // local rotation of gaussian (real, i, j, k)
    hostVertices[i] = g.mean.x;
    hostVertices[i + 1] = g.mean.y;
    hostVertices[i + 2] = g.mean.z;
    hostVertices[i + 3] = g.mean.w;
    hostVertices[i + 4] = g.colour.x;
    hostVertices[i + 5] = g.colour.y;
    hostVertices[i + 6] = g.colour.z;
    hostVertices[i + 7] = g.colour.w;
    hostVertices[i + 8] = g.gid;
    hostVertices[i + 9] = g.scale.x;
    hostVertices[i + 10] = g.scale.y;
    hostVertices[i + 11] = g.scale.z;
    hostVertices[i + 12] = g.rot.x;
    hostVertices[i + 13] = g.rot.y;
    hostVertices[i + 14] = g.rot.z;
    hostVertices[i + 15] = g.rot.w * glm::pi<float>() / 180.0f;
  }
}

// takes a cv::Mat image and returns a copy of the original but with the image partitioned into tiles of size tileHeight x tileWidth.
// dataType is the data type of the image (e.g. CV_8UC3 for 8-bit unsigned char 3-channel image)
// It treats the image as one vector of pixels which we pick from in chunks of tileHeight x tileWidth
cv::Mat tileImageBuffer(cv::Mat image, int tileHeight, int tileWidth, int dataType, int channels) {
    cv::Mat new_image(image.rows, image.cols, dataType);
    uchar *buffer = image.data;
    int stripSize = tileHeight * tileWidth;
     
    for (int j = 0; j < int(floor(image.rows / float(tileHeight))); j++) {
        for (int i = 0; i < int(floor(image.cols / float(tileWidth))); i++) {
            cv::Mat chunk(tileHeight, tileWidth, dataType, buffer, cv::Mat::AUTO_STEP);
            chunk.copyTo(new_image(cv::Rect(i * tileWidth, j * tileHeight, tileWidth, tileHeight)));
            buffer += stripSize * channels;
        }
    }

    return new_image;
}

void IpuSplatter::getFrameBuffer(cv::Mat &frame) const {
  // need to ensure that we read sections of the framebuffer
  // as square tiles and then stitch them together

  cv::Mat image_f = cv::Mat(cv::Size(fbMapping.width, fbMapping.height), CV_32FC4, (void *) frameBuffer.data(), cv::Mat::AUTO_STEP);
  // Clamp values between 0 and 255

  cv::Mat image_f_8u;
  cv::min(image_f * 255.0f, 255.0f, image_f);
  image_f.convertTo(image_f_8u, CV_8UC4);
  cvtColor(image_f_8u, frame, cv::COLOR_RGBA2BGR);
  frame = tileImageBuffer(frame, fbMapping.tileHeight, fbMapping.tileWidth, CV_8UC3, 3);

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

struct MappingInfo {
  std::size_t padding;
  std::size_t elementsPerTile;
  std::size_t totalTiles;
};

MappingInfo calculateMapping(poplar::Graph& g, std::size_t numElements, std::size_t grainSize, TiledFramebuffer &fbMapping) {
  ipu_utils::logger()->info("Input size of pts: {}B", numElements);
  const double numTiles = g.getTarget().getNumTiles();

  if (fbMapping.numTiles < numTiles) {
    ipu_utils::logger()->info("Number of tiles in framebuffer ({}) is less than number of tiles on target ({})", fbMapping.numTiles, numTiles);
  }

  double grainsPerTile = std::ceil(numElements / (fbMapping.numTiles * grainSize));
  double elementsPerTile = grainsPerTile * grainSize;
  double fullTiles = std::floor(numElements / elementsPerTile);
  double remainingElements = numElements - (fullTiles * elementsPerTile);
  double paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;

  fullTiles = fullTiles > 1439 ? 1439 : fullTiles;

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
void addProjectionVertex(poplar::Graph& g, poplar::ComputeSet& cs, unsigned t, const poplar::Tensor& tid,  const poplar::Tensor& westIn, const poplar::Tensor& eastOut,
                         const poplar::Tensor& modelViewProjection, const poplar::Tensor& ptsIn, const poplar::Tensor& localFb) {
  auto v = g.addVertex(cs, "Transform4x4");
  g.setTileMapping(v, t);

  g.connect(v["matrix"], modelViewProjection);
  g.connect(v["vertsIn"], ptsIn);
  g.connect(v["localFb"], localFb);
  g.connect(v["tile_id"], tid);
  g.connect(v["westIn"], westIn);
  g.connect(v["eastOut"], eastOut);

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
  const auto tileMapping = std::string(POPC_PREFIX) + "/include/tileMapping";
  const auto includes = " -I " + glmPath + " -I " + otherIncludes + " -I " + tileMapping;
  ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
  vg.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);

  // Create storage for the model view projeciton matrix. Place the master copy on tile 0
  // and then broadcast from their to all other tiles before any computations.
  modelViewProjection.buildTensor(vg, FLOAT, {4, 4});
  vg.setTileMapping(modelViewProjection, 0u);

  // Build a program to upload and broadcast the modelling-projection matrix:
  program::Sequence broadcastMvp;
  broadcastMvp.add(modelViewProjection.buildWrite(vg, true));

  auto fbGrainSize = 4;
  auto fbToTileMapping = calculateMapping(vg, frameBuffer.size(), fbGrainSize, fbMapping);
  printf("Framebuffer layout: %lu, %lu, %lu\n", fbToTileMapping.padding, fbToTileMapping.elementsPerTile, fbToTileMapping.totalTiles);
  auto paddedFramebuffer = vg.addVariable(FLOAT, {frameBuffer.size() + fbToTileMapping.padding}, "padded_frame_buffer");
  applyTileMapping(vg, paddedFramebuffer, fbToTileMapping);
  
  outputFramebuffer = paddedFramebuffer.slice(0, frameBuffer.size());

  // Map the point cloud vertices across all tiles. TODO: If we are not using AMP the only constraint
  // is that the grain size must be a multiple of 4 (so that 4-vectors are not split between
  // tiles). If we use the AMP we need to have at least 8 4-vectors to fill the AMP pipeline so
  // the minimum grain size is 32:
  const auto grainSize = 4; //disableAMPVertices ? 4 : 4 * 8;

  // MappingInfo{padding, std::size_t(elementsPerTile), std::size_t(fullTiles)};

  printf("hostvertices size: %lu, numtiles %f\n", hostVertices.size(), fbMapping.numTiles);
  auto numElemsPerTile = std::floor(hostVertices.size() / fbMapping.numTiles); // lower bound
  auto remainingElements = hostVertices.size() - (numElemsPerTile * fbMapping.numTiles);
  auto paddedRemainder = std::ceil(remainingElements / grainSize) * grainSize;
  auto padding = paddedRemainder - remainingElements;

  printf("numElemsPerTile: %f, remainingElements : %f, padding: %f\n", numElemsPerTile, remainingElements, padding);
  auto mapping = MappingInfo{std::size_t(padding), std::size_t(numElemsPerTile), std::size_t(fbMapping.numTiles)};
  
  mapping.totalTiles = mapping.totalTiles > 1439 ? 1439 : mapping.totalTiles;

  // auto mapping = calculateMapping(vg, hostVertices.size(), grainSize, fbMapping);
  printf("Vertex layout: %lu, %lu, %lu\n", mapping.padding, mapping.elementsPerTile, mapping.totalTiles);
  auto paddedInput = vg.addVariable(FLOAT, {hostVertices.size() + mapping.padding}, "padded_verts_in");
  applyTileMapping(vg, paddedInput, mapping);

  // We only want to stream to a slice of the padded tensor:
  inputVertices = paddedInput.slice(0, hostVertices.size());

  // Build a compute set to transform the points:
  const auto csName = disableAMPVertices ? "project" : "project_amp";
  auto splatCs = vg.addComputeSet(csName);

  // Get the tile mapping and connect the vertices:
  const auto tm = vg.getTileMapping(paddedInput);
  const auto tmFb = vg.getTileMapping(paddedFramebuffer);

  unsigned numPoints = 100;
  std::size_t channelSize = numPoints * GAUSSIAN_SIZE;

  std::vector<poplar::VertexRef> vertices;

  for (auto t = 0u; t < tm.size(); ++t) {
    const auto& m = tm[t];
    const auto& mFb = tmFb[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected fb to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      // Add the tile local MVP matrix variable and append a copies that broadcast it to all tiles:
      auto localMvp = vg.clone(modelViewProjection, "mvp_tile_" + std::to_string(t));
      vg.setTileMapping(localMvp, t);
      broadcastMvp.add(program::Copy(modelViewProjection, localMvp));

      auto ptsIn = paddedInput.slice(m.front());

      Tensor tid = vg.addConstant<int>(INT, {1}, {int(t)});
      vg.setTileMapping(tid, t);

      auto sliceFb = paddedFramebuffer.slice(mFb.front());

      auto stored = vg.addVariable(poplar::FLOAT, {channelSize});
      vg.setTileMapping(stored, t);

      auto v = vg.addVertex(splatCs, "Transform4x4");
      vg.setTileMapping(v, t);
      vg.connect(v["matrix"], localMvp.flatten());
      vg.connect(v["vertsIn"], ptsIn);
      vg.connect(v["localFb"], sliceFb);
      vg.connect(v["tile_id"], tid);
      vg.connect(v["stored"], stored);
      vertices.push_back(v);
    }
  }

  struct edge r2l("rightOut", "leftIn"); // -->
  struct edge l2r("leftOut", "rightIn"); // <--

  struct edge l2l("leftOut", "leftIn"); // <->
  struct edge r2r("rightOut", "rightIn"); // >-<

  struct edge u2d("upOut", "downIn");
  struct edge d2u("downOut", "upIn");
  
  struct edge u2u("upOut", "upIn");
  struct edge d2d("downOut", "downIn");

  EdgeBuilder eb(vg, vertices, channelSize);

  for (auto t = 0u; t < vertices.size(); ++t) {
    const auto& m = tm[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected fb to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      auto tileOnBoundary = fbMapping.checkImageBoundaries(t);

      if (tileOnBoundary.up) {
        eb.addEdge(t, t, u2u);
      }

      if (tileOnBoundary.left) {
        eb.addEdge(t, t, l2l);
      }

      if (tileOnBoundary.right) {
        eb.addEdge(t, t, r2r);
      }

      if (tileOnBoundary.down) {
        eb.addEdge(t, t, d2d);
      }

      if (!tileOnBoundary.right && t + 1 < vertices.size()) {
        eb.addEdge(t, t + 1, r2l);
        eb.addEdge(t + 1, t, l2r);
      } else if (!tileOnBoundary.right) {
        eb.addEdge(t, t, r2r);
      }

      if (!tileOnBoundary.down && t + fbMapping.numTilesAcross < vertices.size()) {
        eb.addEdge(t, t + fbMapping.numTilesAcross, d2u);
        eb.addEdge(t + fbMapping.numTilesAcross, t, u2d);
      } else if (!tileOnBoundary.down) {
        eb.addEdge(t, t, d2d);
      }
    }
  }

  // this program sequence will copy the points between all the tiles in the graph
  program::Sequence broadcastPoints = eb.getBroadcastSequence();

  program::Sequence main;
  main.add(broadcastMvp);
  main.add(broadcastPoints);
  // main.add(inputVertices.buildWrite(vg, true));
  main.add(program::Execute(splatCs));
  main.add(outputFramebuffer.buildRead(vg, true));

  program::Sequence setup;
  main.add(inputVertices.buildWrite(vg, true));

  getPrograms().add("write_verts", setup);
  getPrograms().add("project", main);
}

void IpuSplatter::execute(poplar::Engine& engine, const poplar::Device& device) {
  if (!initialised) {
    initialised = true;
    modelViewProjection.connectWriteStream(engine, transformMatrix);
    inputVertices.connectWriteStream(engine, hostVertices);
    outputFramebuffer.connectReadStream(engine, frameBuffer);
    getPrograms().run(engine, "write_verts");
  }
  getPrograms().run(engine, "project");
}

} // end of namespace splat
