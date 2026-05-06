#pragma once
#include "passes/semantics_inferer_pass/semantics_inferer.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class ElementwiseBinOperationInferer : public passes::semantics_inferer::SemanticsInferer {
	public:
		auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void override;
	};
}