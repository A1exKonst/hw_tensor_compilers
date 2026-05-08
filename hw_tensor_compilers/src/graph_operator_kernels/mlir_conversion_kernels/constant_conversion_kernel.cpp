#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "graph_operator_kernels/mlir_conversion_kernels/constant_conversion_kernel.h"

#include "mlir/IR/Value.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto ConstantConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        // TODO : add code, specific to given kernel

        return result;
	}
}