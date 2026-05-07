#pragma once
#include <unordered_map>
#include <vector>

#include "graph/graph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"



namespace passes::mlir_conversion {

    [[nodiscard]] 
    auto tranform_graph(mlir::MLIRContext& context_, const graph_engine::Graph& graph_) -> mlir::OwningOpRef<mlir::ModuleOp>;

    auto datatype_to_mlir_type(mlir::OpBuilder& builder, const graph_engine::DataType dtype) -> mlir::Type;

    auto get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, graph_engine::ValueID value_id) -> mlir::RankedTensorType;

    auto get_function_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph) -> mlir::FunctionType;

    auto matmul(mlir::Value a, mlir::Value b, mlir::OpBuilder& builder, mlir::Location loc, bool transpose_b) -> mlir::Value;

    auto scalar_mul(mlir::Value a, float s, mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Value;
};