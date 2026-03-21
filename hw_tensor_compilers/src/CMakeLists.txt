# ============ Core library =================================
add_library(tensor_compiler_lib STATIC
    "graph/graph.cpp"
    "graph/semantics.cpp" 
    "io/in_graph_onnx.cpp"
    "io/out_graph_console.cpp"
    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(tensor_compiler_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}" # currently is /src/
    "${CMAKE_BINARY_DIR}"         # generated protobuf files
)

target_link_libraries(tensor_compiler_lib PUBLIC onnx_proto)

# ============ Main Application ===============================
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PRIVATE tensor_compiler_lib)
