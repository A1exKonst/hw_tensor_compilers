#pragma once
#include "graph/graph.h"
#include "passes/graph_pass.h"

namespace passes
{
	class [[deprecated("Use passes::SemanticsInfererPass instead")]] SemanticsInferer : public GraphPass {
	public:
		SemanticsInferer() {};
		SemanticsInferer(const SemanticsInferer& other) = default;
		SemanticsInferer(SemanticsInferer&& other) = default;

		~SemanticsInferer() = default;

		SemanticsInferer& operator=(const SemanticsInferer&) = default;
		SemanticsInferer& operator=(SemanticsInferer&&) = default;

		auto transform_graph(graph_engine::Graph& graph) -> void override;
	};
}