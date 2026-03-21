cmake_minimum_required(VERSION 3.15)
project(hw_tensor_compilers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============ Dependencies: ==================================

add_subdirectory(hw_tensor_compilers/deps/protobuf)

# ========================== Google Tests: =====================

find_package(GTest CONFIG REQUIRED)

enable_testing()

add_executable(my_tests 
    "hw_tensor_compilers/src/io/in_graph_onnx.cpp"
    "hw_tensor_compilers/src/io/out_graph_console.cpp"
    "hw_tensor_compilers/src/graph/graph.cpp"
    "hw_tensor_compilers/tests/test_main.cpp"
    "hw_tensor_compilers/tests/test_shape.cpp" 
    "hw_tensor_compilers/tests/test_onnx_import.cpp"
    ${PROTO_SRCS}   
    ${PROTO_HDRS}
    #"hw_tensor_compilers/src/io/in_graph_onnx.cpp"
)

target_include_directories(my_tests PRIVATE 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/src"
    "${CMAKE_CURRENT_BINARY_DIR}" # для onnx.pb.h
)

target_link_libraries(my_tests PRIVATE GTest::gtest_main)
target_link_libraries(my_tests PRIVATE onnx_proto)

include(GoogleTest)
gtest_discover_tests(my_tests)

# ============ add_executable: ================================

set(SOURCES
    "hw_tensor_compilers/include/graph/attributes.hpp"
    "hw_tensor_compilers/src/graph/graph.cpp"
    "hw_tensor_compilers/src/graph/semantics.cpp" 

    "hw_tensor_compilers/src/io/in_graph_onnx.cpp"
    "hw_tensor_compilers/src/io/out_graph_console.cpp"

    "hw_tensor_compilers/src/main.cpp"
)

add_executable(${PROJECT_NAME} 
    ${SOURCES}
    ${PROTO_SRCS} 
    ${PROTO_HDRS})

target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/src"
    "${CMAKE_CURRENT_BINARY_DIR}" # generated onnx.pb.h is here
)

# Link with Protobuf
target_link_libraries(${PROJECT_NAME} PRIVATE onnx_proto)

# Setting for Visual Studio (file groups)
source_group("Generated" FILES ${PROTO_SRCS} ${PROTO_HDRS})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
else()
    message(WARNING "Folder data not found! Could not be copied to binary dir.")
endif()
