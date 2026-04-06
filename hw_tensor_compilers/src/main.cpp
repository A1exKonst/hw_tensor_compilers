#include <iostream>
#include <string>
#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"
#include "passes/semantics.h"

#include "passes/mlir_converter.h"
#include "mlir/IR/MLIRContext.h"

//#include "llvm/Support/raw_ostream.h"


int main() {

	try {
		std::string version = "0.1.97";
		std::cout << "exec " << version << std::endl;

		std::string filename = "data/tiny.onnx";
		graph_engine::Graph graph = io::import_from_model(filename);
		std::cout << graph << std::endl;

		semantics::decorate_graph(graph);
		std::cout << graph << std::endl;

		mlir::MLIRContext context;
		passes::GraphToMLIRConverter converter{ context, graph };
		mlir::OwningOpRef<mlir::ModuleOp> tiny_model = converter.convert();

		std::cout << "Module is " << (mlir::succeeded(tiny_model->verify()) ? "VALID" : "INVALID") << std::endl;

		std::cout << "exec " << version << std::endl;
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	};


	return 0;
};