# ============ Core library =================================

set(GRAPH_SOURCES
    "graph/datatype.cpp"
    "graph/shape.cpp" 
    "graph/graph.cpp"
)

set(PASSES_SOURCES
    "passes/mlir_converter.cpp"
    "passes/llvm_mlir_management/set_pipeline.cpp" 
    "passes/llvm_mlir_management/set_context.cpp"
    "passes/mlir_pipeline.cpp"
    "passes/passes_pipeline.cpp"
    "passes/semantics_inferer_pass/semantics_inferer_pass.cpp"
    "passes/semantics_inferer_pass/utils.cpp"
    "passes/mlir_conversion_utils.cpp"
    "passes/mlir_conversion_pass/mlir_conversion_data.cpp"
    "passes/mlir_conversion_pass/mlir_conversion_pass.cpp"
)

set(SEMANTICS_KERNELS
    "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/conv_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/gemm_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/matmul_inferer.cpp"
    "graph_operator_kernels/semantics_inferer_kernels/relu_inferer.cpp"
    "graph_operator_kernels/semantics_inferer_kernels/constant_inferer.cpp"
)

set(MLIR_CONVERSION_KERNELS_SOURCES
    "graph_operator_kernels/mlir_conversion_kernels/add_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/constant_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/conv_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/gemm_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/mul_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/relu_conversion_kernel.cpp"
)

set(KERNELS_SOURCES ${SEMANTICS_KERNELS} ${MLIR_CONVERSION_KERNELS_SOURCES})

set(IO_SOURCES
    "io/onnx_importer.cpp"
    "io/console_graph_exporter.cpp"
)

add_library(tensor_compiler_lib STATIC
    ${GRAPH_SOURCES}
    ${PASSES_SOURCES}
    ${KERNELS_SOURCES}
    ${IO_SOURCES}
    
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
