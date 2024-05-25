

#include <tileMapping/edge_builder.hpp>

namespace splat {

EdgeBuilder::EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize) 
                            : graph(vg), vertexRefs(vertices), channelSize(channelSize) {}

void EdgeBuilder::addBidirectionalEdge(unsigned tid1, unsigned tid2, struct edge e1, struct edge e2) {
    addEdge(tid1, tid2, e1);
    addEdge(tid2, tid1, e2);
}

void EdgeBuilder::addEdge(unsigned tid1, unsigned tid2, struct edge edge) {
    poplar::Tensor outT1 = graph.addVariable(poplar::FLOAT, {channelSize});
    poplar::Tensor inT2 = graph.addVariable(poplar::FLOAT, {channelSize});

    graph.setTileMapping(outT1, tid1);
    graph.setTileMapping(inT2, tid2);

    auto [src, dst] = edge;

    auto v1 = vertexRefs[tid1];
    graph.connect(v1[src], outT1);

    auto v2 = vertexRefs[tid2];
    graph.connect(v2[dst], inT2);

    broadcastSequence.add(poplar::program::Copy(outT1, inT2));
}

void EdgeBuilder::constructLattice(const poplar::Graph::TileToTensorMapping& tm, const TiledFramebuffer& fb) {
    
  struct edge r2l("rightOut", "leftIn"); // -->
  struct edge l2r("leftOut", "rightIn"); // <--

  struct edge l2l("leftOut", "leftIn"); // <->
  struct edge r2r("rightOut", "rightIn"); // >-<

  struct edge u2d("upOut", "downIn");
  struct edge d2u("downOut", "upIn");
  
  struct edge u2u("upOut", "upIn");
  struct edge d2d("downOut", "downIn");

  for (auto t = 0u; t < vertexRefs.size(); ++t) {
    const auto& m = tm[t];
    if (m.size() > 1u) {
      throw std::runtime_error("Expected fb to be stored as a single contiguous region per tile.");
    }
    if (m.size() > 0u) {
      auto tileOnBoundary = fb.checkImageBoundaries(t);

      if (tileOnBoundary.up) {
        addEdge(t, t, u2u);
      }
      if (tileOnBoundary.left) {
        addEdge(t, t, l2l);
      }
      if (tileOnBoundary.right) {
        addEdge(t, t, r2r);
      }
      if (tileOnBoundary.down) {
        addEdge(t, t, d2d);
      }

      if (!tileOnBoundary.right && t + 1 < vertexRefs.size()) {
        addEdge(t, t + 1, r2l);
        addEdge(t + 1, t, l2r);
      } else if (!tileOnBoundary.right) {
        addEdge(t, t, r2r);
      }
      if (!tileOnBoundary.down && t + fb.numTilesAcross < vertexRefs.size()) {
        addEdge(t, t + fb.numTilesAcross, d2u);
        addEdge(t + fb.numTilesAcross, t, u2d);
      } else if (!tileOnBoundary.down) {
        addEdge(t, t, d2d);
      }
    }
  }
}

} // end of namespace splat