#pragma once
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/graph_engine.h"
#include "io/io.h"
#include "passes/passes.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"



TEST(MLIRExecution, TensorDescriptorInitialization_1) {
	graph_engine::Shape s = graph_engine::Shape::make_shape({ 10 });
	std::vector<float> data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
	graph_engine::Tensor<float> tensor = graph_engine::Tensor<float>::make_tensor(data, std::move(s));
	StridedMemRefType<float, 1> descriptor = passes::llvm_mlir_management::make_descriptor<float, 1>(tensor);

	EXPECT_EQ(descriptor.basePtr, tensor.data().data());
	EXPECT_EQ(descriptor.data, tensor.data().data());
	EXPECT_EQ(descriptor.offset, 0);
	EXPECT_EQ(descriptor.sizes[0], 10);
	EXPECT_EQ(descriptor.strides[0], 1);
}

TEST(MLIRExecution, TensorDescriptorInitialization_2) {
	graph_engine::Shape s = graph_engine::Shape::make_shape({ 2, 5 });
	std::vector<float> data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9 };
	graph_engine::Tensor<float> tensor = graph_engine::Tensor<float>::make_tensor(data, std::move(s));
	StridedMemRefType<float, 2> descriptor = passes::llvm_mlir_management::make_descriptor<float, 2>(tensor);

	EXPECT_EQ(descriptor.basePtr, tensor.data().data());
	EXPECT_EQ(descriptor.data, tensor.data().data());
	EXPECT_EQ(descriptor.offset, 0);
	EXPECT_EQ(descriptor.sizes[0], 2);
	EXPECT_EQ(descriptor.sizes[1], 5);
	EXPECT_EQ(descriptor.strides[0], 5);
	EXPECT_EQ(descriptor.strides[1], 1);
}

TEST(MLIRExecution, TensorDescriptorInitialization_3) {
	graph_engine::Shape s = graph_engine::Shape::make_shape({ 2, 3, 2 });
	std::vector<float> data{ 0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11 };
	graph_engine::Tensor<float> tensor = graph_engine::Tensor<float>::make_tensor(data, std::move(s));
	StridedMemRefType<float, 3> descriptor = passes::llvm_mlir_management::make_descriptor<float, 3>(tensor);

	EXPECT_EQ(descriptor.basePtr, tensor.data().data());
	EXPECT_EQ(descriptor.data, tensor.data().data());
	EXPECT_EQ(descriptor.offset, 0);
	EXPECT_EQ(descriptor.sizes[0], 2);
	EXPECT_EQ(descriptor.sizes[1], 3);
	EXPECT_EQ(descriptor.sizes[2], 2);
	EXPECT_EQ(descriptor.strides[0], 6);
	EXPECT_EQ(descriptor.strides[1], 2);
	EXPECT_EQ(descriptor.strides[2], 1);
}

