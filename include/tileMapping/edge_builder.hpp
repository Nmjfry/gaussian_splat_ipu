
#pragma once

#include <cstdlib>

#include <ipu/ipu_utils.hpp>
#include <poputil/TileMapping.hpp>
#include <tileMapping/tile_config.hpp>
#include <string>
#include <queue>

namespace splat {

struct socketDesc {
    std::string in;
    std::string out;

    socketDesc(std::string in, std::string out) : in(in), out(out) {}

    static struct socketDesc getLeftSocket() {
      return {"leftOut", "leftIn"};
    }
    static struct socketDesc getRightSocket() {
      return {"rightOut", "rightIn"};
    }
    static struct socketDesc getUpSocket() {
      return {"upOut", "upIn"};
    }
    static struct socketDesc getDownSocket() {
      return {"downOut", "downIn"};
    }
};

struct edgeDesc {
    struct socketDesc t1;
    struct socketDesc t2;

    edgeDesc(struct socketDesc t1, struct socketDesc t2) : t1(t1), t2(t2) {}
};

struct edge {
  std::string src;
  std::string dst;
  edge(std::string src, std::string dst) : src(src), dst(dst) {}
};

class EdgeBuilder{
public:
  EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize, TiledFramebuffer& fb);
  virtual ~EdgeBuilder() {}

  // std::pair<directions, directions> getFreeNeighbouringEdges(unsigned tid);
  
  void addBidirectionalEdge(unsigned tid1, unsigned tid2, struct edge e1, struct edge e2);
  void addEdge(unsigned tid1, unsigned tid2, struct edge edge);
  poplar::program::Sequence getBroadcastSequence() { return broadcastSequence; }

private:
    poplar::Graph& graph;
    std::vector<poplar::VertexRef>& vertexRefs;
    std::size_t channelSize;
    // TODO: channelType (Char)
    TiledFramebuffer& fbMapping;
    // keeps a record of the edges that have been added <tileID, <directions>>
    std::unordered_map<unsigned, std::pair<directions, directions>> existingEdges;

    std::vector<std::vector<poplar::Tensor>> tileChannels;
    poplar::program::Sequence broadcastSequence;

};

} // end of namespace splat
