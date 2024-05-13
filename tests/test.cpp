// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <ipu/ipu_utils.hpp>
#include </home/nf20/workspace/gaussian_splat_ipu/include/tileMapping/tile_config.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

ipu_utils::RuntimeConfig testConfig {
  1, 1, // numIpus, numReplicas
  "ipu_test", // exeName
  false, false, false, // useIpuModel, saveExe, loadExe
  false, true // compileOnly, deferredAttach
};

BOOST_AUTO_TEST_CASE(Glm) {
  float m[] = {1, 2, 3, 4,
               5, 6, 7, 8,
               9, 10, 11, 12,
               13, 14, 15, 16};
  float v[] = {2, 4, 6, 8};
  const auto mgl = glm::transpose(glm::make_mat4(m));
  auto vgl = glm::make_vec4(v);
  vgl = mgl * vgl;
  BOOST_CHECK_EQUAL(vgl.x, 60);
  BOOST_CHECK_EQUAL(vgl.y, 140);
  BOOST_CHECK_EQUAL(vgl.z, 220);
  BOOST_CHECK_EQUAL(vgl.w, 300);
}

// Use a lambda builder to run an isolated and stateless vertex
// (e.g. a vertex that runs a self contained test). Returns the
// exit code of the vertex runner (equal to EXIT_SUCCESS
// if the vertex ran with no errors).
int runStatelessVertex(std::string vertexName) {
  using namespace poplar;

  auto ipuTest = ipu_utils::LambdaBuilder(
    // Build test graph:
    [&](Graph& graph, const Target& target, ipu_utils::ProgramManager& progs) {
      const auto codeletFile = std::string(POPC_PREFIX) + "/codelets/tests/codelets.cpp";
      const auto glmPath = std::string(POPC_PREFIX) + "/external/glm/";
      const auto otherIncludes = std::string(POPC_PREFIX) + "/include/missing";
      const auto includes = " -I " + glmPath + " -I " + otherIncludes;
      ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
      graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3" + includes);

      auto cs1 = graph.addComputeSet("test_cs");
      auto v1 = graph.addVertex(cs1, vertexName);
      graph.setTileMapping(v1, 0);

      program::Sequence runTest({program::Execute(cs1)});
      progs.add("test", runTest);
    },
    // Run test graph:
    [&](Engine& engine, const Device& device, const ipu_utils::ProgramManager& progs) {
      progs.run(engine, "test");
    }
  );

  ipuTest.setRuntimeConfig(testConfig);
  return ipu_utils::GraphManager().run(ipuTest);
}

BOOST_AUTO_TEST_CASE(IpuGlm) {
  using namespace poplar;
  spdlog::set_level(spdlog::level::warn);

  BOOST_CHECK_EQUAL(EXIT_SUCCESS, runStatelessVertex("TfbBoundsCheck"));
  BOOST_CHECK_EQUAL(EXIT_SUCCESS, runStatelessVertex("GlmMat4"));
  BOOST_CHECK_EQUAL(EXIT_SUCCESS, runStatelessVertex("GlmTransform"));
}
