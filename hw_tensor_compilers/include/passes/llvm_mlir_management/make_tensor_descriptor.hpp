#pragma once
#include "graph/tensor.hpp"

#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "mlir/IR/BuiltinOps.h"

namespace passes::llvm_mlir_management {

	template<typename DType, int N>
	StridedMemRefType<DType, N> make_descriptor(graph_engine::Tensor<DType>& tensor) {

        if (N != tensor.shape().rank()) {
            throw std::length_error("make_descriptor(...) : incompatible Shape and StridedMemRefType::N");
        }

        StridedMemRefType<DType, N> descriptor;

        // row-major: stride[i] = size[i+1] * size[i+2] * ...
        auto shape = tensor.shape();
        size_t current_stride = 1;
        for (int i = N - 1; i >= 0; --i) {
            descriptor.sizes[i] = shape[i];
            descriptor.strides[i] = current_stride;
            current_stride *= shape[i];
        }

        auto* data_ptr = tensor.data().data();
        descriptor.basePtr = data_ptr;
        descriptor.data = data_ptr;
        descriptor.offset = 0;

        return descriptor;
	};

}