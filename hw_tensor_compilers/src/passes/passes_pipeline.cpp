#include <iostream>

#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "io/io.h"

// PipelineEndpoint::EXECUTION dependencies:
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "llvm/Support/TargetSelect.h"


namespace passes {
	void PassesPipeline::apply_pipeline(const std::string& filename, passes::PipelineEndpoint endpoint, bool debug) {
		std::cout << "================ onnx -> graph ====================================" << std::endl;
		graph_engine::Graph graph = io::import_from_model(filename);
		std::cout << graph << std::endl;
		if (endpoint == PipelineEndpoint::GRAPH_INPUT) return;

		std::cout << "================ semantics(graph) ====================================" << std::endl;
		SemanticsInferer::transform_graph(graph);
		std::cout << graph << std::endl;
		if (endpoint == PipelineEndpoint::SEMANTICS_INFERER) return;

		std::cout << "================ graph -> mlir ====================================" << std::endl;
		mlir::MLIRContext context;
		passes::llvm_mlir_management::set_context(context);
		mlir::OwningOpRef<mlir::ModuleOp> model = passes::GraphToMLIRConverter::tranform_graph(context, graph);
		std::cout << "Module is " << (mlir::succeeded(model->verify()) ? "valid" : "INVALID") << std::endl;
		model->dump();
		if (endpoint == PipelineEndpoint::MLIR_GENERATION) return;

		std::cout << "================ mlir -> llvm ====================================" << std::endl;
		passes::MLIRPipeline::lower_to_llvm(*model, debug);
		std::cout << std::endl;
		model->dump();
		if (endpoint == PipelineEndpoint::MLIR_LOWERING) return;

		std::cout << "================ exec(llvm) ====================================" << std::endl;
		llvm::InitializeNativeTarget();
		llvm::InitializeNativeTargetAsmPrinter();
		llvm::InitializeNativeTargetAsmParser();
		mlir::ExecutionEngineOptions options;
		//options.jitCodeGenOptLevel = llvm::CodeGenOpt::Aggressive;

		llvm::SmallVector<llvm::StringRef, 4> runtimeLibs;
		runtimeLibs.push_back("libmlir_runner_utils.so");
		runtimeLibs.push_back("libmlir_c_runner_utils.so");

		options.sharedLibPaths = runtimeLibs;
		auto engine_result = mlir::ExecutionEngine::create(*model);
		std::cout << "ExecutionEngine created: " << bool(engine_result) << std::endl;
		if (!bool(engine_result)) {
			llvm::errs() << "Engine creation failed: " << engine_result.takeError() << "\n";
		}
		std::unique_ptr<mlir::ExecutionEngine> engine = std::move(engine_result.get());

		graph_engine::Shape input_s = graph_engine::Shape::make_shape({ 1, 10 });
		std::vector<float> input_data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
		graph_engine::Tensor<float> input_tensor = graph_engine::Tensor<float>::make_tensor(input_data, std::move(input_s));
		StridedMemRefType<float, 2> input_descriptor = passes::llvm_mlir_management::make_descriptor<float, 2>(input_tensor);

		StridedMemRefType<float, 2> result_placeholder;

		std::cout << "descriptors created" << std::endl;
		
		StridedMemRefType<float, 2>* input_ptr = &input_descriptor;

		void* args[] = { &result_placeholder, &input_descriptor };
		auto error = engine->invokePacked("main", args);
		if (error) {
			llvm::errs() << "Execution failed\n";
		}
		

		if (endpoint == PipelineEndpoint::EXECUTION) return;

		/*
		graph_engine::Shape output_s = graph_engine::Shape::make_shape({ 1, 10 });
		std::vector<float> output_data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
		graph_engine::Tensor<float> output_tensor = graph_engine::Tensor<float>::make_tensor(input_data, std::move(output_s));
		StridedMemRefType<float, 2> output_descriptor = passes::llvm_mlir_management::make_descriptor<float, 2>(output_tensor);
		*/
		
	};
}