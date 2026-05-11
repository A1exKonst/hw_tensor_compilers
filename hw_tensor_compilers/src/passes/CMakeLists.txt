set(PASSES_SOURCES
    
    "passes/mlir_management/set_pipeline.cpp" 
    "passes/mlir_management/set_context.cpp"
    "passes/mlir_management/lower_to_llvm.cpp"
    "passes/passes_pipeline.cpp"

    "passes/semantics_inferer_pass/semantics_inferer_pass.cpp"
    "passes/semantics_inferer_pass/utils.cpp"

    "passes/mlir_conversion_pass/utils.cpp"
    "passes/mlir_conversion_pass/mlir_conversion_data.cpp"
    "passes/mlir_conversion_pass/mlir_conversion_pass.cpp"

    PARENT_SCOPE
)