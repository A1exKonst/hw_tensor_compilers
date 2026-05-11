set(SEMANTICS_KERNELS
    "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/conv_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/gemm_inferer.cpp" 
    "graph_operator_kernels/semantics_inferer_kernels/matmul_inferer.cpp"
    "graph_operator_kernels/semantics_inferer_kernels/relu_inferer.cpp"
    "graph_operator_kernels/semantics_inferer_kernels/constant_inferer.cpp"
)

set(MLIR_CONVERSION_KERNELS
    "graph_operator_kernels/mlir_conversion_kernels/add_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/constant_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/conv_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/gemm_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/mul_conversion_kernel.cpp" 
    "graph_operator_kernels/mlir_conversion_kernels/relu_conversion_kernel.cpp"
)

set(KERNELS_SOURCES 
    ${SEMANTICS_KERNELS} 
    ${MLIR_CONVERSION_KERNELS}
    PARENT_SCOPE
)