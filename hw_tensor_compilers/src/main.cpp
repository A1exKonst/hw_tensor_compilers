#include <iostream>
#include <string>

#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"
#include "passes/semantics.h"

int main() {

	try {
		std::string version = "0.1.82";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/single_conv.onnx";
		graph_engine::Graph graph = io::import_from_model(filename);
		std::cout << graph << std::endl;

		semantics::decorate_graph(graph);
		std::cout << graph << std::endl;

		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};