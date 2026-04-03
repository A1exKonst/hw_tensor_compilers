# ============ Core library =================================
add_library(tensor_compiler_lib STATIC
    "graph/datatype.cpp"
    "graph/shape.cpp" 
    "graph/graph.cpp"

    "passes/semantics.cpp" 
    "passes/mlir_converter.cpp"

    "io/in_graph_onnx.cpp"
    "io/out_graph_console.cpp"

    "mlir_handler.cpp"
    "mlir_handler.h"

    ${PROTO_SRCS}
    ${PROTO_HDRS}
)

target_include_directories(tensor_compiler_lib PUBLIC 
    "${PROJECT_SOURCE_DIR}/hw_tensor_compilers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}" # currently is /src/
    "${CMAKE_BINARY_DIR}"         # generated protobuf files
    #${MLIR_INCLUDE_DIRS}
    #${LLVM_INCLUDE_DIRS}
)

target_link_libraries(tensor_compiler_lib PUBLIC 
    onnx_proto
    mlir_deps
)
get_target_property(TC_LIBS tensor_compiler_lib INTERFACE_LINK_LIBRARIES)
message(STATUS "Full link libraries for tensor_compiler_lib: ${TC_LIBS}")

# ============ Main Application ===============================
message(STATUS "CHECK: LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} PUBLIC tensor_compiler_lib)
