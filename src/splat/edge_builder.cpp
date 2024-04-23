

#include <tileMapping/edge_builder.hpp>

namespace splat {

EdgeBuilder::EdgeBuilder(poplar::Graph& vg, std::vector<poplar::VertexRef>& vertices, std::size_t channelSize, TiledFramebuffer& fb) 
                            : graph(vg), vertexRefs(vertices), channelSize(channelSize), fbMapping(fb) {
    // Empty constructor
    // tileChannels.resize(fbMapping.numTiles);
    // for (auto i = 0; i < fbMapping.numTiles; i++) {
    //     tileChannels[i].resize(fbMapping.numTiles);
    // }
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

void EdgeBuilder::generateLocalConnectivity(unsigned tid) {
    auto [inDirs, outDirs] = getFreeNeighbouringEdges(tid);
    auto vertex = vertexRefs[tid];

    // if (outDirs.right) {
    // printf("tid: %d, right\n", tid);
    // auto westIn = tileInChannels[tid];
    // auto eastOut = tileOutChannels[tid];
    // graph.connect(vertex["westIn"], westIn);
    // graph.connect(vertex["eastOut"], eastOut);
    // if (tid > 0) {
    //     auto eOut = tileOutChannels[tid - 1];
    //     broadcastSequence.add(poplar::program::Copy(eOut, westIn));
    // }
    // // }
}

void EdgeBuilder::generateTileChannels() {

    // for (auto i = 0; i < fbMapping.numTiles; i++) {
    //     auto v = graph.addVariable(poplar::FLOAT, {channelSize});
    //     graph.setTileMapping(v, tid);
    // }

    // auto eastOut = graph.addVariable(poplar::FLOAT, {channelSize});
    // auto westIn = graph.addVariable(poplar::FLOAT, {channelSize});
    // graph.setTileMapping(westIn, tid);
    // graph.setTileMapping(eastOut, tid);
    // tileInChannels[tid] = westIn;
    // tileOutChannels[tid] = eastOut;

    // if (outDirs.left) {
    //     auto westOut = vg.addVariable(FLOAT, {channelSize});
    //     auto eastIn = vg.addVariable(FLOAT, {channelSize});
    //     vg.setTileMapping(westOut, tid);
    //     vg.setTileMapping(eastIn, tid - 1);
    //     vg.connect(vertex["westOut"], westOut);
    //     vg.connect(vertex["eastIn"], eastIn);
    // }
    // if (outDirs.down) {
    //     auto southOut = vg.addVariable(FLOAT, {channelSize});
    //     auto northIn = vg.addVariable(FLOAT, {channelSize});
    //     vg.setTileMapping(southOut, tid);
    //     vg.setTileMapping(northIn, tid + fbMapping.numTilesAcross);
    //     vg.connect(vertex["southOut"], southOut);
    //     vg.connect(vertex["northIn"], northIn);
    // }
    // if (outDirs.up) {
    //     auto northOut = vg.addVariable(FLOAT, {channelSize});
    //     auto southIn = vg.addVariable(FLOAT, {channelSize});
    //     vg.setTileMapping(northOut, tid);
    //     vg.setTileMapping(southIn, tid - fbMapping.numTilesAcross);
    //     vg.connect(vertex["northOut"], northOut);
    //     vg.connect(vertex["southIn"], southIn);
    // }

}



void EdgeBuilder::addLocalOutEdges(unsigned tid, directions dirs) {
    // Empty function
}

void EdgeBuilder::addLocalInEdges(unsigned tid, directions dirs) {
    // Empty function
}

} // end of namespace splat