#include <iostream>
#include <string>

#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"

#include "passes/mlir_converter.h"
#include "passes/semantics_inferer.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"



int main() {

	try {
		std::string version = "0.1.105";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/tiny.onnx";
		graph_engine::Graph graph = io::import_from_model(filename);
		std::cout << graph << std::endl;

		passes::SemanticsInferer::transform_graph(graph);
		std::cout << graph << std::endl;
		
		mlir::MLIRContext context;
		passes::GraphToMLIRConverter converter{ context, graph };
		mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

		std::cout << "Module is " << (mlir::succeeded(model->verify()) ? "VALID" : "INVALID") << std::endl;
		
		
		mlir::PassManager pm{model->getContext()};

		pm.addPass(mlir::createConvertFuncToLLVMPass());
		pm.addPass(mlir::createConvertControlFlowToLLVMPass());

		pm.addPass(mlir::createConvertTensorToLinalgPass());
		pm.addPass(mlir::createConvertLinalgToStandardPass());
		pm.addPass(mlir::createArithToLLVMConversionPass());

		pm.addPass(mlir::createConvertMemRefToSPIRVPass());
		//pm.addPass(mlir::createReconcileUnrealizedCastsPass());

		// pm.run(*model);
		
		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};