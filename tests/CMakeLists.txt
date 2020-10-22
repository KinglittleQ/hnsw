add_executable(test test.cpp)
target_include_directories(test PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(sift_test sift_test.cpp)
target_include_directories(sift_test PUBLIC "${PROJECT_SOURCE_DIR}/include")
