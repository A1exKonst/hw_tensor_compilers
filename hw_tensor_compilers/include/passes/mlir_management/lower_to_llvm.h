#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/MLIRContext.h"



namespace passes::mlir_management {

	[[nodiscard]]
	auto lower_to_llvm(mlir::ModuleOp model, bool ir_printing) -> mlir::LogicalResult;

}