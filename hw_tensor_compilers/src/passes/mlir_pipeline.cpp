#include "passes/mlir_pipeline.h"
#include "mlir/IR/BuiltinOps.h"

namespace passes {
	auto MLIRPipeline::lower_to_llvm(mlir::ModuleOp model, bool ir_printing) -> mlir::LogicalResult {
		mlir::PassManager pm{ model->getContext() };
		llvm_mlir_management::set_pipeline(pm);

		if (ir_printing) {
			pm.getContext()->disableMultithreading();
			pm.enableIRPrinting();
		}

		auto result = pm.run(model);
		return result;
	};
}