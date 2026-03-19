# TODO:
# 0. add command to copy data/


cmake_minimum_required(VERSION 3.15)
project(hw_tensor_compilers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============ Dependencies: ==================================

# =========================== Protobuf: =======================

# Protobuf is required for onnx.pb.h (input of .onnx files)
find_package(Protobuf REQUIRED)

set(PROTO_INPUT_DIR "hw_tensor_compilers/deps/protobuf")
set(PROTO_SRC "${PROTO_INPUT_DIR}/onnx.proto")

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS "${PROTO_SRC}")

# ========================== Google Tests: =====================

find_package(GTest CONFIG REQUIRED)

enable_testing()

add_executable(my_tests 
    "hw_tensor_compilers/src/io/in_graph_onnx.cpp"
    "hw_tensor_compilers/src/io/out_graph_console.cpp"
    "hw_tensor_compilers/src/graph/graph.cpp"
    "hw_tensor_compilers/tests/test_main.cpp"
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
target_link_libraries(my_tests PRIVATE protobuf::libprotobuf)

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
    ${PROTO_HDRS}
)

target_include_directories(${PROJECT_NAME} 
    PRIVATE 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/src"
    "${CMAKE_CURRENT_BINARY_DIR}" # generated onnx.pb.h is here
)

# Link with Protobuf
target_link_libraries(${PROJECT_NAME} PRIVATE protobuf::libprotobuf)

# Setting for Visual Studio (file groups)
source_group("Generated" FILES ${PROTO_SRCS} ${PROTO_HDRS})

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
else()
    message(WARNING "Folder data not found! Could not be copied to binary dir.")
endif()


# add_custom_command(
#    TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#        "${CMAKE_CURRENT_SOURCE_DIR}/data"
#        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/data"
#)
