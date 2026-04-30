#include <iostream>
#include <string>

#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"

#include "passes/mlir_converter.h"
#include "passes/semantics_inferer.h"
#include "passes/mlir_pipeline.h"

#include "passes/llvm_mlir_management/llvm_mlir_management.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "llvm/Support/TargetSelect.h"


int main() {

	try {
		std::string version = "0.2.35";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/single_relu.onnx";
		graph_engine::Graph graph = io::import_from_model(filename);
		std::cout << graph << std::endl;

		passes::SemanticsInferer::transform_graph(graph);
		std::cout << graph << std::endl;
		
		mlir::MLIRContext context;
		passes::llvm_mlir_management::set_context(context);

		mlir::OwningOpRef<mlir::ModuleOp> model = passes::GraphToMLIRConverter::tranform_graph(context, graph);
		std::cout << "Module is " << (mlir::succeeded(model->verify()) ? "valid" : "INVALID") << std::endl;
		model->dump();

		passes::MLIRPipeline::lower_to_llvm(*model);
		std::cout << std::endl;
		model->dump();

		llvm::InitializeNativeTarget();
		llvm::InitializeNativeTargetAsmPrinter();
		llvm::InitializeNativeTargetAsmParser();
		auto engine = mlir::ExecutionEngine::create(*model);
		std::cout << "ExecutionEngine created: " << bool(engine) << std::endl;
		if (!bool(engine)) {
			llvm::errs() << "Engine creation failed: " << engine.takeError() << "\n";
		}
		/*
		float inputData[10] = { 0, 1, -2, 3, -4, 5, -6, 7, -8, 9 };
		float outputData[10] = { 0 };

		std::string funcName = "main";
		void* args[] = { &inputData, &outputData };
		auto error = engine->get()->invokePacked(funcName, args);
		if (error) {
			llvm::errs() << "Execution failed: " << error << "\n";
		}
		*/
		//engine(inputData, outputData);
		
		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};