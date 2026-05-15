#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/relu_inferer_kernel.h"
#include "graph/graph.h"
#include "passes/semantics_inferer_pass/utils.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto ReluInfererKernel::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);
		expect(node.outputs.size() == 1, "Node[Relu] : one output Value is expected");
		expect(node.inputs.size() == 1, "Node[Relu] : one input Value is expected");
		expect_dtype(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).dtype); // expect equal dtypes
		expect_shape(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).shape); // expect equal shapes

		return;
	}

}