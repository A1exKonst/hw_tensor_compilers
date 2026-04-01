#include "mlir_handler.h"

namespace my_mlir {
	void test_print() {
		mlir::MLIRContext context;
		std::cout << "MLIR Context created successfully! " << 
			context.isMultithreadingEnabled() << " " <<
			context.allowsUnregisteredDialects() << std::endl;
	}
}
