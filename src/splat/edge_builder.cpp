

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

} // end of namespace splat