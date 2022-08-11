link_libraries(gtest_main gmock)

add_executable(utils_test utils_test.cc)
target_link_libraries(utils_test PUBLIC utils)
# add_test(<name> <command> [<arg>...])
add_test(utils_test utils_test)

# test case wrong
add_executable(ctc_prefix_beam_search_test ctc_prefix_beam_search_test.cc)
target_link_libraries(ctc_prefix_beam_search_test PUBLIC decoder utils)
add_test(ctc_prefix_beam_search_test ctc_prefix_beam_search_test)
set_tests_properties(ctc_prefix_beam_search_test PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${PADDLE_LIB_DIRS}:{$LD_LIBRARY_PATH}")


add_executable(feature_pipeline_test feature_pipeline_test.cc)
target_link_libraries(feature_pipeline_test PUBLIC utils frontend)
# add_test(<name> <command> [<arg>...])
add_test(feature_pipeline_test feature_pipeline_test)