#pragma once
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace passes::mlir_management {

	void set_context(mlir::MLIRContext& context);

}