#pragma once
#include "passes/semantics_inferer_pass/semantics_inferer.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class ReluInferer : public passes::semantics_inferer::SemanticsInferer {
	public:
		ReluInferer() = default;

		auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void override;

	};

}