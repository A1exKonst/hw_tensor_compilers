#include "passes/llvm_mlir_management/set_pipeline.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
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
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
//#include "mlir/Dialect/Tensor/IR/TensorTilingInterface.h" 

namespace passes::llvm_mlir_management {

	void set_pipeline(mlir::PassManager& pm) {

		// 1. tensor.empty -> bufferization.alloc_tensor
		pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

		// 2. bufferization (allocation of all tensors):
		mlir::bufferization::OneShotBufferizationOptions options;
		options.bufferizeFunctionBoundaries = true;
		options.setFunctionBoundaryTypeConversion(
			mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap);
		options.allowUnknownOps = true;
		pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));

		// 3. bufferization (deallocation, remove memory leaks):
		mlir::bufferization::BufferDeallocationPipelineOptions deallocOptions;
		mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocOptions);

		pm.addPass(mlir::createConvertLinalgToLoopsPass());				// 4. linalg -> scf.parallel, scf.for
		pm.addPass(mlir::createConvertSCFToCFPass());					// 5. scf -> cf (basic_blocks, branches)
		pm.addPass(mlir::createArithToLLVMConversionPass());			// 6. arith.addi -> llvm.add
		pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());	// 7. memref -> llvm struct descriptors
		pm.addPass(mlir::createConvertControlFlowToLLVMPass());			// 8. cf -> llvm.br, llvm.cond_br
		//mlir::LowerToLLVMOptions llvm_options(&context);
		//llvm_options.emitCWrappers = true;
		//pm.addPass(mlir::createAddCFuncWrapperPass());
		pm.addPass(mlir::createConvertFuncToLLVMPass());				// 9. mlir.func -> llvm.func. Changes types in func signature
		pm.addPass(mlir::createReconcileUnrealizedCastsPass());			// 20.remove builtin.unrealized_conversion_cast
	}

}