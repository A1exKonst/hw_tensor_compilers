#pragma once
#include <unordered_map>

#include "graph/graph.h"
#include "graph/node.h"
#include "passes/graph_pass.h"
#include "passes/semantics_inferer_pass/semantics_inferer.h"



namespace passes
{
	class SemanticsInfererPass : public GraphPass {
	public:
		SemanticsInfererPass(); // fill registry_
		SemanticsInfererPass(const SemanticsInfererPass& other) = default;
		SemanticsInfererPass(SemanticsInfererPass&& other) = default;

		~SemanticsInfererPass() = default;

		SemanticsInfererPass& operator=(const SemanticsInfererPass&) = default;
		SemanticsInfererPass& operator=(SemanticsInfererPass&&) = default;

		auto transform_graph(graph_engine::Graph& graph) -> void override;

	private:
		std::unordered_map<
			graph_engine::OperatorType, 
			passes::semantics_inferer::SemanticsInferer
		> registry_;
	};
}