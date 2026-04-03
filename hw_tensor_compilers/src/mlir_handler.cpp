#include "mlir_handler.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace my_mlir {
	void test_print() {
		mlir::MLIRContext context;
		mlir::Value v;
		std::cout << "MLIR Context created successfully! " << 
			context.isMultithreadingEnabled() << " " <<
			context.allowsUnregisteredDialects() << std::endl;
	}
}
