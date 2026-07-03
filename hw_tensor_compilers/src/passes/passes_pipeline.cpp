#include <iostream>

#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "io/onnx_importer.h"
#include "io/console_graph_exporter.h"

// PipelineEndpoint::EXECUTION dependencies:
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"


#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Debug.h"
#include <cstring>



/*
extern "C" {
	__declspec(dllexport) void memrefCopy(int64_t elementSize,
		void* sBase, void* sPtr, int64_t sOff,
		void* dBase, void* dPtr, int64_t dOff,
		int64_t size) {
		char* src = (char*)sPtr + sOff * elementSize;
		char* dst = (char*)dPtr + dOff * elementSize;
		if (size > 0) {
			std::memcpy(dst, src, size * elementSize);
		}
	}
}
*/

static auto my_jit_copy_stub(
	int64_t elemSize, void* s1, void* s2, int64_t s3,
	void* d1, void* d2, int64_t d3, int64_t size
) -> void {
	// minimal support for JIT
	std::memcpy(d2, s2, size * elemSize);
	return;
}


namespace passes {

	auto PassesPipeline::apply_pipeline(passes::PipelineEndpoint endpoint, bool debug) -> void {
		std::cout << "================ onnx -> graph ====================================" << std::endl;
		graph_engine::Graph graph = importer.import_graph();
		exporter << graph;
		if (endpoint == PipelineEndpoint::GRAPH_INPUT) return;

		std::cout << "================ passes(graph) ====================================" << std::endl;
		for (const auto& pass : graph_passes) {
			pass->transform_graph(graph);
			exporter << graph;
		}

		if (endpoint == PipelineEndpoint::GRAPH_PASSES) return;

		std::cout << "================ graph -> mlir ====================================" << std::endl;
		mlir::MLIRContext context;
		passes::mlir_management::set_context(context);
		mlir::OwningOpRef<mlir::ModuleOp> model = passes::MLIRConversionPass(graph, context).convert();
		std::cout << "Module is " << (mlir::succeeded(model->verify()) ? "valid" : "INVALID") << std::endl;
		model->dump();
		if (endpoint == PipelineEndpoint::MLIR_GENERATION) return;

		std::cout << "================ mlir -> llvm ====================================" << std::endl;
		mlir::LogicalResult is_lowered = passes::mlir_management::lower_to_llvm(*model, debug);
		if (mlir::failed(is_lowered)) return;
		std::cout << std::endl;
		model->dump();
		if (endpoint == PipelineEndpoint::MLIR_LOWERING) return;

		std::cout << "================ exec(llvm) ====================================" << std::endl;
		llvm::InitializeNativeTarget();
		llvm::InitializeNativeTargetAsmPrinter();
		llvm::InitializeNativeTargetAsmParser();

		/*
		std::cout << "> Registering symbols..." << std::endl;
		void* ptr = reinterpret_cast<void*>(&memrefCopy);
		llvm::sys::DynamicLibrary::AddSymbol("memrefCopy", ptr);
		llvm::sys::DynamicLibrary::AddSymbol("_memrefCopy", ptr);
		llvm::sys::DynamicLibrary::AddSymbol("_mlir_ciface_memrefCopy", ptr);

		llvm::DebugFlag = true;
		*/
		std::cout << "> create ExecutionEngine" << std::endl;
		mlir::ExecutionEngineOptions options;

		auto engine_result = mlir::ExecutionEngine::create(*model);
		std::cout << "ExecutionEngine created: " << bool(engine_result) << std::endl;
		if (!bool(engine_result)) {
			llvm::errs() << "Engine creation failed: " << engine_result.takeError() << "\n";
		}
		std::unique_ptr<mlir::ExecutionEngine> engine = std::move(engine_result.get());

		graph_engine::Shape input_s = graph_engine::Shape::make_shape({ 1, 10 });
		std::vector<float> input_data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
		graph_engine::Tensor<float> input_tensor = graph_engine::Tensor<float>::make_tensor(input_data, std::move(input_s));
		StridedMemRefType<float, 2> input_descriptor = passes::mlir_management::make_descriptor<float, 2>(input_tensor);

		StridedMemRefType<float, 2> result_placeholder;

		std::cout << "descriptors created" << std::endl;
		
		StridedMemRefType<float, 2>* input_ptr = &input_descriptor;

		void* args[] = { &result_placeholder, &input_descriptor };
		std::cout << "call engine->invoke(main)" << std::endl;
		auto error = engine->invokePacked("_mlir_ciface_main", args);
		if (error) {
			llvm::errs() << "Execution failed\n";
		}
		llvm::DebugFlag = false;

		if (endpoint == PipelineEndpoint::EXECUTION) return;

		/*
		graph_engine::Shape output_s = graph_engine::Shape::make_shape({ 1, 10 });
		std::vector<float> output_data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
		graph_engine::Tensor<float> output_tensor = graph_engine::Tensor<float>::make_tensor(input_data, std::move(output_s));
		StridedMemRefType<float, 2> output_descriptor = passes::llvm_mlir_management::make_descriptor<float, 2>(output_tensor);
		*/
		
	}

}