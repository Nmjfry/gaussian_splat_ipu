
#pragma once

#include <cstdlib>

#include <ipu/ipu_utils.hpp>
#include <poputil/TileMapping.hpp>
#include <tileMapping/tile_config.hpp>
#include <string>

namespace splat {

struct edge {
  std::string src;
  std::string dst;
  edge(std::string src, std::string dst) : src(src), dst(dst) {}
  void print() {
    printf("%s --> %s\n", src.c_str(), dst.c_str());
  }

};

class EdgeBuilder{
public:
  EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize);
  virtual ~EdgeBuilder() {}
  
  void addBidirectionalEdge(unsigned tid1, unsigned tid2, struct edge e1, struct edge e2);
  void addEdge(unsigned tid1, unsigned tid2, struct edge edge);
  poplar::program::Sequence getBroadcastSequence() { return broadcastSequence; }

private:
    poplar::Graph& graph;
    std::vector<poplar::VertexRef>& vertexRefs;
    std::size_t channelSize;
    // TODO: channelType (Char)
    poplar::program::Sequence broadcastSequence;

};

} // end of namespace splat
