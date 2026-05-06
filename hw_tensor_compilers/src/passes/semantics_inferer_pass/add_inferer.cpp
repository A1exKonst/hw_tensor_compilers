#pragma once
#include "passes/semantics_inferer_pass/add_inferer.h"
#include "graph/graph.h"

namespace passes::semantics_inferer {

	auto AddInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
	};
}