cmake_minimum_required(VERSION 3.15)
project(hw_tensor_compilers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# folder /bin for all executables and files, used by executables:
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# ============ Dependencies ==================================
add_subdirectory(hw_tensor_compilers/deps/protobuf)

option(BUILD_TESTING "Build tests" ON)

# ============ Core library =================================
add_library(tensor_compiler_lib STATIC
    "hw_tensor_compilers/src/graph/graph.cpp"
    "hw_tensor_compilers/src/graph/semantics.cpp" 
    "hw_tensor_compilers/src/io/in_graph_onnx.cpp"
    "hw_tensor_compilers/src/io/out_graph_console.cpp"
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(tensor_compiler_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/src"
    "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(tensor_compiler_lib PUBLIC onnx_proto)

# ============ Main Application ===============================
add_executable(${PROJECT_NAME} "hw_tensor_compilers/src/main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE tensor_compiler_lib)

# ============ Tests =========================================
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(hw_tensor_compilers/tests)
endif()

# copy data from project to executable:
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/hw_tensor_compilers/data" 
            DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()
