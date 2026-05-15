#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "graph/node.h"
#include "passes/graph_pass.h"
#include "passes/semantics_inferer_pass/semantics_inferer_kernel.h"



namespace passes
{
	class SemanticsInfererPass : public GraphPass {
	public:
		SemanticsInfererPass(); // fill registry_
		~SemanticsInfererPass() = default;

		SemanticsInfererPass(const SemanticsInfererPass& other) = default;
		SemanticsInfererPass(SemanticsInfererPass&& other) = default;

		SemanticsInfererPass& operator=(const SemanticsInfererPass&) = default;
		SemanticsInfererPass& operator=(SemanticsInfererPass&&) = default;

		auto transform_graph(graph_engine::Graph& graph) -> void override;

	private:
		std::unordered_map<
			graph_engine::OperatorType, 
			std::unique_ptr<passes::semantics_inferer::SemanticsInfererKernel>
		> registry_;
	};
}