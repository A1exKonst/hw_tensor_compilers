#pragma once
#include "graph/graph.h"

namespace passes {

	namespace semantics_inferer {

		/*
		* Interface class for inference of different graph/node.h/OperatorType
		*/
		class SemanticsInferer {
		public:
			virtual ~SemanticsInferer() = default;
			SemanticsInferer(const SemanticsInferer&) = delete;
			SemanticsInferer& operator=(const SemanticsInferer&) = delete;

			auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void = 0;

		protected:
			SemanticsInferer() = default;
		};

	}
}