#pragma once
#include <unordered_map>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "graph/graph.h"

namespace passes {

    class GraphToMLIRConverter {
    private:
        const graph_engine::Graph& graph;

        mlir::MLIRContext& context;

        mlir::OpBuilder builder;

        std::unordered_map<graph_engine::ValueID, mlir::Value> value_id_to_mlir_value;

        auto convert_graph_value_to_mlir_recursively(graph_engine::ValueID value) -> mlir::Value;

        template <typename IntOp, typename FloatOp>
        auto create_binary_operation(graph_engine::NodeID producer) -> mlir::Value;

    public:
        explicit GraphToMLIRConverter(mlir::MLIRContext& context_, const graph_engine::Graph& graph_) :
            context(context_), builder(&context), graph(graph_) {
        };

        auto convert() -> mlir::OwningOpRef<mlir::ModuleOp>;

        static auto datatype_to_mlir_type(mlir::OpBuilder& builder, const graph_engine::DataType dtype) -> mlir::Type;

        static auto get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, graph_engine::ValueID value_id) -> mlir::RankedTensorType;

        static auto get_function_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph) -> mlir::FunctionType;

    };

    auto matmul(mlir::Value a, mlir::Value b, mlir::OpBuilder& builder, mlir::Location loc, bool transpose_b) -> mlir::Value;

    auto scalar_mul(mlir::Value a, float s, mlir::OpBuilder& builder, mlir::Location loc)->mlir::Value;
};