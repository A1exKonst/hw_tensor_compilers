#pragma once
#include <iostream>
#include <string>

#include "io/io.h"
#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "passes/llvm_mlir_management/llvm_mlir_management.h"



int main() {

	try {
		std::string version = "0.2.67";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/single_add.onnx";
		// correct lowering		: gemm, relu, mul, add
		// incorrect lowering	: conv
		// no gen				: matmul 
		
		std::vector<std::unique_ptr<passes::GraphPass>> passes;
		passes.push_back(std::make_unique<passes::SemanticsInfererPass>());

		auto importer = io::OnnxImporter(filename);
		auto exporter = io::ConsoleGraphExporter();

		passes::PassesPipeline pipeline(
			importer,
			exporter,
			std::move(passes)
		);

		pipeline.apply_pipeline(passes::PipelineEndpoint::EXECUTION);
		
		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};