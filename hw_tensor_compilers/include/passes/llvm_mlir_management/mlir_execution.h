#pragma once
#include "graph/tensor.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/IR/BuiltinOps.h"

namespace passes::llvm_mlir_management {

	template<typename DType>
	graph_engine::Tensor<DType> execute(graph_engine::Tensor<DType>& input, mlir::ModuleOp& model, mlir::ExecutionEngine& engine);

	template<typename DType>
	graph_engine::Tensor<DType> execute(graph_engine::Tensor<DType>& input, mlir::ModuleOp& model);

	template<typename DType, size_t N>
	mlir::StridedMemRefType<DType, N> create_descriptor(graph_engine::Tensor<DType>& t);

}