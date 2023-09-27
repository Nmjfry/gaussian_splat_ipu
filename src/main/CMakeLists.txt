
message(STATUS "Adding splat executable: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(splat splat.cpp)
target_link_libraries(splat
  Boost::program_options
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  ${POPLAR_LINK_LIBRARIES}
  splat_utils
)