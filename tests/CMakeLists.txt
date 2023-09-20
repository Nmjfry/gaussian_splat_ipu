
add_executable(tests test.cpp)
target_link_libraries(tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${POPLAR_LINK_LIBRARIES})
add_test(test1 tests)
