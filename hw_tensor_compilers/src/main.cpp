#pragma once
#include <iostream>
#include <string>

#include "io/io.h"
#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "passes/llvm_mlir_management/llvm_mlir_management.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "llvm/Support/TargetSelect.h"


int main() {

	try {
		std::string version = "0.2.37";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/single_add.onnx";
		// correct lowering: gemm, relu
		// incorrect lowering : mul, add
		// no gen: matmul, conv
		
		passes::PassesPipeline::apply_pipeline(filename, passes::PipelineEndpoint::MLIR_GENERATION);
		
		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};