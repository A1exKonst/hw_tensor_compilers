#pragma once
#include "graph/graph.h"



namespace passes {

	namespace semantics_inferer {

		/*
		* Interface class for inference of different graph/node.h/OperatorType
		*/
		class SemanticsInfererKernel {
		public:
			virtual ~SemanticsInfererKernel() = default;
			SemanticsInfererKernel(const SemanticsInfererKernel&) = delete;
			SemanticsInfererKernel& operator=(const SemanticsInfererKernel&) = delete;

			virtual auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void = 0;

		protected:
			SemanticsInfererKernel() = default;
		};

	}
}