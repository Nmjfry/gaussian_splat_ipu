

#include <tileMapping/edge_builder.hpp>

namespace splat {

EdgeBuilder::EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize, TiledFramebuffer& fb) 
                            : graph(vg), vertexRefs(vertices), channelSize(channelSize), fbMapping(fb) {

}


std::pair<directions, directions> EdgeBuilder::getFreeNeighbouringEdges(unsigned tid) {
    directions availableInDirs = fbMapping.checkImageBoundaries(tid);
    directions availableOutDirs = availableInDirs;

    auto unsetUsedDirs = [](directions& dirs, directions existingDirs) {
        dirs.left = dirs.left && !existingDirs.left;
        dirs.up = dirs.up && !existingDirs.up;
        dirs.right = dirs.right && !existingDirs.right;
        dirs.down = dirs.down && !existingDirs.down;
    };

    if (existingEdges.find(tid) != existingEdges.end()) {
        // if the edge has already been added, remove it from the available directions
        auto [inDirs, outDirs] = existingEdges[tid];
        unsetUsedDirs(availableInDirs, inDirs);
        unsetUsedDirs(availableOutDirs, outDirs);
    }

    return std::make_pair(availableInDirs, availableOutDirs);    
}

void EdgeBuilder::addBidirectionalEdge(unsigned tid1, unsigned tid2, struct edgeDesc edge) {
    poplar::Tensor outT1 = graph.addVariable(poplar::FLOAT, {channelSize});
    poplar::Tensor inT1 = graph.addVariable(poplar::FLOAT, {channelSize});
    poplar::Tensor outT2 = graph.addVariable(poplar::FLOAT, {channelSize});
    poplar::Tensor inT2 = graph.addVariable(poplar::FLOAT, {channelSize});

    graph.setTileMapping(outT1, tid1);
    graph.setTileMapping(inT1, tid1);
    graph.setTileMapping(outT2, tid2);
    graph.setTileMapping(inT2, tid2);

    auto [out1, in1] = edge.t1;
    auto [out2, in2] = edge.t2;

    auto v1 = vertexRefs[tid1];
    graph.connect(v1[in1], inT1);
    graph.connect(v1[out1], outT1);

    auto v2 = vertexRefs[tid2];
    graph.connect(v2[in2], inT2);
    graph.connect(v2[out2], outT2);

    broadcastSequence.add(poplar::program::Copy(outT1, inT2));
    broadcastSequence.add(poplar::program::Copy(outT2, inT1));
}

} // end of namespace splat