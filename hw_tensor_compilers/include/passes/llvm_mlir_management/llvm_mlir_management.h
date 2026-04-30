#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace passes::llvm_mlir_management {

	void set_context(mlir::MLIRContext& context);

	void set_pipeline(mlir::PassManager& pm);

}