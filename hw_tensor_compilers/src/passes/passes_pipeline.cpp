#include <iostream>

#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "io/onnx_importer.h"
#include "io/console_graph_exporter.h"



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

		if (endpoint == PipelineEndpoint::EXECUTION) return;
		
	}

}