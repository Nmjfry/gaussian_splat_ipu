
message(STATUS "Adding splat library: ${CMAKE_CURRENT_SOURCE_DIR}")

FILE(GLOB_RECURSE LIB_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/splat/*.hpp)

FILE(GLOB_RECURSE LIB_INC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tileMapping/*.hpp)
FILE(GLOB_RECURSE LIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(splat_utils ${LIB_INC} ${LIB_SRC})
