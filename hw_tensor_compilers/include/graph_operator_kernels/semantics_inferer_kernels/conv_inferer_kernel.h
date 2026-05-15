#pragma once
#include "passes/semantics_inferer_pass/semantics_inferer_kernel.h"
#include "graph/graph.h"



namespace passes::semantics_inferer {

	class ConvInfererKernel : public passes::semantics_inferer::SemanticsInfererKernel {
	public:
		ConvInfererKernel() = default;

		auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void override;

	};

}