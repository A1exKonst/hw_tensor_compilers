#pragma once
#include "passes/semantics_inferer_pass/matmul_inferer.h"
#include "graph/graph.h"
#include "passes/semantics_inferer_pass/utils.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto MatmulInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);

		expect(node.outputs.size() == 1,
			"Node[Gemm] : one output Value is expected");
		expect(node.inputs.size() == 2,
			"Node[Gemm] : 3 input Values are expected");

		ValueID out = node.outputs.at(0);
		ValueID first = node.inputs.at(0);
		ValueID second = node.inputs.at(1);

		// expect dtypes:
		DataType result_type = graph_engine::math_result_data_type(graph.values.at(first).dtype, graph.values.at(second).dtype);
		expect_dtype(graph, first, result_type);
		expect_dtype(graph, second, result_type);
		expect_dtype(graph, out, result_type);

		// expect shapes:
		auto result_rank = graph.values[first].shape.rank();
		expect(result_rank == graph.values[second].shape.rank(),
			"Node[MatMul] : equal ranks of input Values are expected");

		Shape result_shape = Shape(result_rank);

		// expect shapes: Broadcasting:
		for (int i = 0; i < result_rank - 2; ++i) {
			expect(graph.values[first].shape[i] == graph.values[second].shape[i],
				"Node[MatMul] : different broadcasted shape dimensions found");
			result_shape[i] = graph.values[first].shape[i];
		}

		// expect shapes: MatMul last 2 dims:
		expect(graph.values[first].shape[result_rank - 1] == graph.values[second].shape[result_rank - 2],
			"Node[MatMul] : tensors last 2 dims (M*N1 and N2*K). N1 == N2 is expected, but it is false");
		result_shape[result_rank - 2] = graph.values[first].shape[result_rank - 2];  // dimension M
		result_shape[result_rank - 1] = graph.values[second].shape[result_rank - 1]; // dimension K

		graph.values[out].shape = std::move(result_shape);
	};

}