#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer.h"
#include "passes/semantics_inferer_pass/utils.h"
#include "graph/graph.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto ElementwiseBinOperationInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);
		expect(node.inputs.size() == 2, "Node[Elementwise Binop] : two input Values are expected");
		expect(node.outputs.size() == 1, "Node[Elementwise Binop] : one output Value is expected");

		ValueID out = node.outputs.at(0);
		ValueID first = node.inputs.at(0);
		ValueID second = node.inputs.at(1);

		DataType result_type = graph_engine::math_result_data_type(
			graph.values.at(first).dtype,
			graph.values.at(second).dtype
		);
		expect_dtype(graph, first, result_type);
		expect_dtype(graph, second, result_type);
		expect_dtype(graph, out, result_type);

		std::optional<Shape> result_shape = graph_engine::calculate_broadcast_compatible_shape(
			graph.values[first].shape,
			graph.values[second].shape
		);
		expect(result_shape.has_value(), "Node[Elementwise Binop] : couldn't broadcast shapes");
		expect_shape(graph, first, result_shape.value());
		expect_shape(graph, second, result_shape.value());
		expect_shape(graph, out, std::move(result_shape.value()));

		return;
	}

}