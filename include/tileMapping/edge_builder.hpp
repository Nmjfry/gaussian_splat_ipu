
#pragma once

#include <cstdlib>

#include <ipu/ipu_utils.hpp>
#include <poputil/TileMapping.hpp>
#include <tileMapping/tile_config.hpp>

namespace splat {

class EdgeBuilder{
public:
  EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize, TiledFramebuffer& fb);
  virtual ~EdgeBuilder() {}

  std::pair<directions, directions> getFreeNeighbouringEdges(unsigned tid);
  void addLocalOutEdges(unsigned tid, directions dirs);
  void addLocalInEdges(unsigned tid, directions dirs);
  void generateLocalConnectivity(unsigned tid); 
  void generateTileChannels();
  poplar::program::Sequence getConnectivity() {
    return broadcastSequence;
  }

private:
    poplar::Graph& graph;
    std::vector<poplar::VertexRef>& vertexRefs;
    std::size_t channelSize;
    // TODO: channelType (Char)
    TiledFramebuffer& fbMapping;
    // keeps a record of the edges that have been added <tileID, <directions>>
    std::unordered_map<unsigned, std::pair<directions, directions>> existingEdges;
    poplar::program::Sequence broadcastSequence;

    std::vector<std::vector<poplar::Tensor>> tileChannels;

};

} // end of namespace splat
