#pragma once
#include <iostream>
#include <string>

#include "io/io.h"
#include "passes/passes.h"
#include "graph/graph_engine.h"
#include "passes/llvm_mlir_management/llvm_mlir_management.h"


int main() {

	try {
		std::string version = "0.2.45";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/single_relu.onnx";
		//graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
		//io::ConsoleGraphExporter out;
		//out << "Hello, GraphExporter!" << std::endl;
		//out << graph;

		// correct lowering		: gemm, relu, mul, add
		// incorrect lowering	: conv
		// no gen				: matmul 
		
		passes::PassesPipeline::apply_pipeline(filename, passes::PipelineEndpoint::SEMANTICS_INFERER);
		
		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};