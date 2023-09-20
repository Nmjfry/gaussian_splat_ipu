
function(get_poplar_version)
  execute_process(COMMAND bash "-c" "popc --version | cut -d ' ' -f3 | head -1" OUTPUT_VARIABLE POPLAR_VERSION)
  string(REPLACE "." ";" VERSION_LIST ${POPLAR_VERSION})
  list(GET VERSION_LIST 0 POPLAR_VERSION_MAJOR)
  list(GET VERSION_LIST 1 POPLAR_VERSION_MINOR)
  list(GET VERSION_LIST 2 POPLAR_VERSION_PATCH)
  set(POPLAR_VERSION_MAJOR ${POPLAR_VERSION_MAJOR} PARENT_SCOPE)
  set(POPLAR_VERSION_MINOR ${POPLAR_VERSION_MINOR} PARENT_SCOPE)
  set(POPLAR_VERSION_PATCH ${POPLAR_VERSION_PATCH} PARENT_SCOPE)
endfunction()


function(check_for_submodules)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/glm/.git")
    message(WARNING "The GLM submodule has not been initialised.")
  endif()
endfunction()