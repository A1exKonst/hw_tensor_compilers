#include "passes/mlir_management/set_pipeline.h"

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
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Transforms/Passes.h"


//#include "mlir/Dialect/Tensor/IR/TensorTilingInterface.h" 

namespace passes::mlir_management {

	auto set_pipeline(mlir::PassManager& pm) -> void {
		// 1. tensor.empty -> bufferization.alloc_tensor
		// 2. bufferization (allocation of all tensors)
		/* 3. Destination-Passing Style:
				mlir is generated with "return tensor<...>",
				but llvm is more stable in destination-passing style.
				especially it is required when using llvm.emit_c_interface
				This pass transforms function signature */
		// 4. bufferization (deallocation, remove memory leaks):
		// 4. linalg -> scf.parallel, scf.for
		// 5. scf -> cf (basic_blocks, branches)
		// 6. arith.addi -> llvm.add
		// 8. cf -> llvm.br, llvm.cond_br
		// 7. memref -> llvm struct descriptors
		// 9. mlir.func -> llvm.func. Changes types in func signature
		// 10.remove builtin.unrealized_conversion_cast

		pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

		mlir::bufferization::OneShotBufferizationOptions options;
		options.bufferizeFunctionBoundaries = true;
		options.setFunctionBoundaryTypeConversion(
			mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
		pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
		pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass()); // remove subview
		pm.addPass(mlir::createCanonicalizerPass());
		pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

		mlir::bufferization::BufferDeallocationPipelineOptions deallocOptions;
		mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocOptions);

		pm.addPass(mlir::createConvertLinalgToLoopsPass());
		pm.addPass(mlir::memref::createNormalizeMemRefsPass());

		pm.addPass(mlir::memref::createExpandStridedMetadataPass());
		pm.addPass(mlir::createCanonicalizerPass());

		pm.addPass(mlir::createConvertSCFToCFPass());
		pm.addPass(mlir::createConvertControlFlowToLLVMPass());
		pm.addPass(mlir::createArithToLLVMConversionPass());
		pm.addPass(mlir::createConvertFuncToLLVMPass());
		//pm.addPass(mlir::createCanonicalizerPass());

		pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
		
		pm.addPass(mlir::createReconcileUnrealizedCastsPass());
		pm.addPass(mlir::createCanonicalizerPass());
		/*
		pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());

		mlir::bufferization::OneShotBufferizationOptions options;
		options.bufferizeFunctionBoundaries = true;
		options.setFunctionBoundaryTypeConversion(
			mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
		pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
		pm.addPass(mlir::memref::createFoldMemRefAliasOpsPass()); // remove subview
		pm.addPass(mlir::createCanonicalizerPass());
		pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());

		mlir::bufferization::BufferDeallocationPipelineOptions deallocOptions;
		mlir::bufferization::buildBufferDeallocationPipeline(pm, deallocOptions);

		pm.addPass(mlir::createConvertLinalgToLoopsPass());
		pm.addPass(mlir::memref::createNormalizeMemRefsPass());

		pm.addPass(mlir::memref::createExpandStridedMetadataPass());
		pm.addPass(mlir::createCanonicalizerPass());

		pm.addPass(mlir::createConvertSCFToCFPass());
		pm.addPass(mlir::createConvertControlFlowToLLVMPass());

		pm.addPass(mlir::createConvertSCFToCFPass());
		pm.addPass(mlir::createCanonicalizerPass());

		pm.addPass(mlir::createArithToLLVMConversionPass());
		pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
		pm.addPass(mlir::createConvertFuncToLLVMPass());
		pm.addPass(mlir::createReconcileUnrealizedCastsPass());
		*/

	}

}