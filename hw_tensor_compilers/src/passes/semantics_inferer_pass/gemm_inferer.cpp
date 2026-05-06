#pragma once
#include "passes/semantics_inferer_pass/gemm_inferer.h"
#include "graph/graph.h"

namespace passes::semantics_inferer {

	auto GemmInferer::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
	};
}