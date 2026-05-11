#pragma once
#include "graph/graph.h"



namespace passes {

	/*
	* Interface class for passes, that modify or process graph_engine::Graph
	*/
	class GraphPass {
	public:
		virtual ~GraphPass() = default;
		GraphPass(const GraphPass&) = delete;
		GraphPass& operator=(const GraphPass&) = delete;

		virtual auto transform_graph(graph_engine::Graph& graph) -> void = 0;

	protected:
		GraphPass() = default;
	};

}