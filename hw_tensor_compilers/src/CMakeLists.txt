# ============ Core library =================================
add_library(tensor_compiler_lib STATIC
    "graph/datatype.cpp"
    "graph/shape.cpp" 
    "graph/graph.cpp"
    
    "passes/mlir_converter.cpp"
    "passes/llvm_mlir_management/set_pipeline.cpp" 
    "passes/llvm_mlir_management/set_context.cpp"
    "passes/mlir_pipeline.cpp"
    "passes/passes_pipeline.cpp"
    
    "passes/semantics_inferer_pass/semantics_inferer_pass.cpp"
    "passes/semantics_inferer_pass/utils.cpp"
    "passes/semantics_inferer_pass/elementwise_binop_inferer.cpp" 
    "passes/semantics_inferer_pass/conv_inferer.cpp" 
    "passes/semantics_inferer_pass/gemm_inferer.cpp" 
    "passes/semantics_inferer_pass/matmul_inferer.cpp"
    "passes/semantics_inferer_pass/relu_inferer.cpp"
    "passes/semantics_inferer_pass/constant_inferer.cpp"

    "io/onnx_importer.cpp"
    "io/console_graph_exporter.cpp"

    ${PROTO_SRCS}
    ${PROTO_HDRS}
 "passes/mlir_conversion_utils.cpp" "passes/mlir_conversion_pass/mlir_conversion_data.cpp")

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
