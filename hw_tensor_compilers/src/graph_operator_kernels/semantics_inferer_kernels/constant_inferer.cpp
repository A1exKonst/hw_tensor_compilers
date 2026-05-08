#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/constant_inferer.h"
#include "graph/graph.h"
#include "passes/semantics_inferer_pass/utils.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto ConstantInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);
		expect(node.outputs.size() == 1, "Node[Constant] : one output Value is expected");
	};

}