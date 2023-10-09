
message(STATUS "Adding splat executable: ${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(splat splat.cpp)
target_link_libraries(splat
  ${PACKETCOMMS_LIBRARIES}
  ${VIDEOLIB_LIBRARIES}
  Boost::program_options
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
  splat_utils
  ${POPLAR_LINK_LIBRARIES}
)
