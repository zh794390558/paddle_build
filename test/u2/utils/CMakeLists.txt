add_library(utils STATIC
    utils.cc
)
target_include_directories(utils PUBLIC ${PROJECT_SOURCE_DIR})