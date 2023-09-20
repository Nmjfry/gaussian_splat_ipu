// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <ipu_utils.hpp>

#include <glm/mat4x4.hpp>
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

BOOST_AUTO_TEST_CASE(IpuGlm) {
  using namespace poplar;
  spdlog::set_level(spdlog::level::warn);

  auto ipuTest = ipu_utils::LambdaBuilder(
    // Build test graph:
    [&](Graph& graph, const Target& target, ipu_utils::ProgramManager& progs) {
      const auto codeletFile = std::string(POPC_PREFIX) + "/codelets/tests/codelets.cpp";
      const auto includePath = std::string(POPC_PREFIX) + "/external/glm/";
      ipu_utils::logger()->debug("POPC_PREFIX: {}", POPC_PREFIX);
      graph.addCodelets(codeletFile, poplar::CodeletFileType::Auto, "-O3 -I " + includePath);

      auto cs1 = graph.addComputeSet("test_cs");
      auto v1 = graph.addVertex(cs1, "GlmMat4");
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
  auto exitCode = ipu_utils::GraphManager().run(ipuTest);
  BOOST_CHECK_EQUAL(exitCode, EXIT_SUCCESS);
}
