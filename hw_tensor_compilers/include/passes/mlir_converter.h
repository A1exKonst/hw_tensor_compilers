#pragma once
#include <unordered_map>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "graph/graph.h"



class GraphToMLIRConverter {
private:
    const graph_engine::Graph& graph;

    mlir::MLIRContext& context;

    std::unordered_map<graph_engine::ValueID, mlir::Value> value_id_to_mlir_value;

    mlir::Value createLinalgGemm(mlir::OpBuilder& b, std::vector<mlir::Value> inputs);
public:
    explicit GraphToMLIRConverter(mlir::MLIRContext& context_, mlir::OpBuilder& builder_, const graph_engine::Graph& graph_) :
        context(context_), graph(graph_) { };

    mlir::OwningOpRef<mlir::ModuleOp> convert() {

        // === init_module() :

        mlir::OpBuilder builder(&context);
        auto loc = builder.getUnknownLoc();

        mlir::ModuleOp module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToStart(module.getBody());

        // === mlir::FunctionType determine_function_type() :

        mlir::FunctionType funcType; // not ready
        // todo: initialize func type
        // auto tensorType = mlir::RankedTensorType::get({2, 2}, builder.getF32Type());
        // llvm::SmallVector<mlir::Type, 1> inputs = { tensorType }; 
        // llvm::SmallVector<mlir::Type, 1> results = { tensorType }; 
        // mlir::FunctionType funcType = builder.getFunctionType(inputs, results);

        auto funcOp = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
        mlir::Block* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        // === void convert_graph_nodes() :

        // mlir::Type datatype_to_mlir_type(graph_engine::DataType dtype);

        //llvm::ArrayRef<int64_t> shapeSlice(fullShape.data(), rank);
        //auto tensorType = mlir::RankedTensorType::get(shapeSlice, builder.getF32Type());

        // todo: add graph.nodes visit
        // OperatorType::CONSTANT -> arith::ConstantOp
        // OperatorType::ADD -> arith::AddFOp, arith::AddIOp
        // OperatorType::MUL -> arith::MulFOp, arith::MulIOp 

        return mlir::OwningOpRef<mlir::ModuleOp>(module);
    };

    mlir::Value convert_nodes_recursively(graph_engine::NodeID node);
};