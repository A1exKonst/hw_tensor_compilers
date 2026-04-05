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
    public:
        explicit GraphToMLIRConverter(mlir::MLIRContext& context_, const graph_engine::Graph& graph_) :
            context(context_), builder(&context), graph(graph_) {
        };

        mlir::OwningOpRef<mlir::ModuleOp> convert() {
            auto loc = builder.getUnknownLoc();

            mlir::ModuleOp module = mlir::ModuleOp::create(loc);
            builder.setInsertionPointToStart(module.getBody());

            mlir::FunctionType funcType = get_function_type(builder, graph);
            auto funcOp = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
            mlir::Block* entryBlock = funcOp.addEntryBlock();
            builder.setInsertionPointToStart(entryBlock);

            // === void convert_graph_nodes() :

            // todo: add graph.nodes visit
            // OperatorType::CONSTANT -> arith::ConstantOp
            // OperatorType::ADD -> arith::AddFOp, arith::AddIOp
            // OperatorType::MUL -> arith::MulFOp, arith::MulIOp



            return mlir::OwningOpRef<mlir::ModuleOp>(module);
        };

        auto convert_value_to_mlir_value(graph_engine::NodeID node) -> mlir::Value;

        template <typename IntOp, typename FloatOp>
        auto create_binary_operation(graph_engine::NodeID producer)->mlir::Value;

        static auto datatype_to_mlir_type(mlir::OpBuilder& builder, const graph_engine::DataType dtype) -> mlir::Type;

        static auto get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, graph_engine::ValueID value_id) -> mlir::RankedTensorType;

        static auto get_function_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph) -> mlir::FunctionType;

    };

};