#include "passes/mlir_management/lower_to_llvm.h"
#include "passes/mlir_management/set_context.h"
#include "passes/mlir_management/set_pipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/MLIRContext.h"



namespace passes::mlir_management {

	auto lower_to_llvm(mlir::ModuleOp model, bool ir_printing) -> mlir::LogicalResult {
		mlir::PassManager pm{ model->getContext() };
		mlir_management::set_pipeline(pm);

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