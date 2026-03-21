find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(my_tests 
    "test_main.cpp"
    "test_shape.cpp" 
    "test_onnx_import.cpp"
)

target_link_libraries(my_tests 
    PRIVATE 
    tensor_compiler_lib
    GTest::gtest_main
)

gtest_discover_tests(my_tests)
