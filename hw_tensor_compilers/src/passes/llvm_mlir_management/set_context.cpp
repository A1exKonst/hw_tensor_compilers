#include "passes/llvm_mlir_management/set_context.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
//#include "mlir/Dialect/Tensor/IR/TensorTilingInterface.h" 

namespace passes::llvm_mlir_management {

	void set_context(mlir::MLIRContext& context) {
		mlir::DialectRegistry registry;
		registry.insert<
			mlir::linalg::LinalgDialect,					// linalg dialect
			mlir::arith::ArithDialect,						// arith dialect
			mlir::tensor::TensorDialect,					// tensor dialect
			mlir::memref::MemRefDialect,					// allocation of tensors in memory; MemRef{shape, element_type, layout_map, memory_space}
			mlir::scf::SCFDialect,							// linalg representation in structured cycles. keeps information about parallel cycles : scf.parallel
			mlir::bufferization::BufferizationDialect,		// a process of conversion from tensor to memref
			mlir::func::FuncDialect,						// regions, arguments, return values
			mlir::LLVM::LLVMDialect							// conversion from mlir to llvm, with help of translators, such as llvm.mlir.constant
		>();

		// Instructions on how to do bufferization:
		mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);					// which args of linalg.generic need or do not need a separate buffer
		mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);					// conversion to memref from i.e. tensor.slice
		mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
		mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);	// bufferization of function args and return values
		mlir::registerBuiltinDialectTranslation(registry);		// mlir translation
		mlir::registerLLVMDialectTranslation(registry);			// llvm translation

		context.appendDialectRegistry(registry);
		context.loadAllAvailableDialects();
	}

}