#include <iostream>
#include <string>
#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"
#include "passes/semantics.h"
#include "mlir_handler.h"

//#include "llvm/Support/raw_ostream.h"
//#include "mlir/IR/MLIRContext.h"

int main() {

	try {
		std::string version = "0.1.97";
		std::cout << "exec " << version << std::endl;

		my_mlir::test_print();

		std::string filename = "data/tiny.onnx";
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