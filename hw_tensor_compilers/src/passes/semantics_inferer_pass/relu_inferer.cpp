#pragma once
#include "passes/semantics_inferer_pass/relu_inferer.h"
#include "graph/graph.h"
#include "passes/semantics_inferer_pass/utils.h"



namespace passes::semantics_inferer {

	auto ReluInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
	};

}