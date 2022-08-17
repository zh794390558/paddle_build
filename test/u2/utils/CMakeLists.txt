add_library(utils STATIC
    utils.cc
    string.cc
)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR})