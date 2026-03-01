#include <iostream>
#include <string>

#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"
#include "graph/semantics.h"

int main() {

	try {
		std::cout << "test 0.1.36" << std::endl;

		std::string filename = "data/tiny.onnx";

		graph_engine::Graph graph = io::import_from_model(filename);

		semantics::decorate_graph(graph);

		std::cout << graph << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};