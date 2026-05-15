#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/gemm_inferer_kernel.h"
#include "passes/semantics_inferer_pass/utils.h"
#include "graph/graph.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto GemmInfererKernel::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);

		// expect input output arguments amount:
		expect(node.outputs.size() == 1, "Node[Gemm] : one output Value is expected");
		expect(node.inputs.size() == 3, "Node[Gemm] : 3 input Values are expected");


		// expect dtypes:
		ValueID out = node.outputs.at(0);
		ValueID first = node.inputs.at(0);
		ValueID second = node.inputs.at(1);
		ValueID third = node.inputs.at(2);

		Value& first_val = graph.values.at(first);
		Value& second_val = graph.values.at(second);
		Value& third_val = graph.values.at(third);

		DataType result_type = graph_engine::math_result_data_type(
			graph_engine::math_result_data_type(
				first_val.dtype,
				second_val.dtype
			),
			third_val.dtype
		);

		expect_dtype(graph, first, result_type);
		expect_dtype(graph, second, result_type);
		expect_dtype(graph, third, result_type);
		expect_dtype(graph, out, result_type);


		// expect shapes:
		expect(first_val.shape.rank() == 2 && second_val.shape.rank() == 2,
			"Values for Gemm : only rank == 2 allowed");

		first_val.shape.rank(2);
		second_val.shape.rank(2);

		int64_t is_transposed_B = std::get<int64_t>(graph.nodes[node_id].attr.at("transB"));
		std::optional<Shape> matmul_shape;
		if (!is_transposed_B) {
			matmul_shape = graph_engine::calculate_matmul_compatible_shape(first_val.shape, second_val.shape);
		}
		else {
			matmul_shape = graph_engine::calculate_matmul_compatible_shape(first_val.shape, transposed(second_val.shape));
		}
		expect(matmul_shape.has_value(), "Values for Gemm : cannot multiply matrices");

		std::optional<Shape> gemm_shape = graph_engine::calculate_broadcast_compatible_shape(matmul_shape.value(), third_val.shape);
		expect(gemm_shape.has_value(), "Values for Gemm : cannot add matrices");
		expect_shape(graph, out, gemm_shape.value());

		return;
	}

}