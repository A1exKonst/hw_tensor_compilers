#include "passes/mlir_pipeline.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/MLIRContext.h"


namespace passes {
	auto MLIRPipeline::lower_to_llvm(mlir::ModuleOp model, bool ir_printing) -> mlir::LogicalResult {
		mlir::PassManager pm{ model->getContext() };
		llvm_mlir_management::set_pipeline(pm);

		if (ir_printing) {
			pm.getContext()->disableMultithreading();
			pm.enableVerifier(true);
			pm.enableIRPrinting(
				/*shouldPrintBeforePass=*/[](mlir::Pass*, mlir::Operation*) { return true; },
				/*shouldPrintAfterPass=*/[](mlir::Pass*, mlir::Operation*) { return true; },
				/*printAfterOnlyOnError=*/false,
				/*printAfterOnlyOnChanges=*/false,
				/*printModuleScope=*/true,
				llvm::errs()
			);

		}

		auto result = pm.run(model);
		return result;
	};
}