#pragma once
#include "passes/llvm_mlir_management/llvm_mlir_management.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"

namespace passes {
	class MLIRPipeline {
	public:
		static auto lower_to_llvm(mlir::ModuleOp model, bool ir_printing = false) -> mlir::LogicalResult;
	};
}