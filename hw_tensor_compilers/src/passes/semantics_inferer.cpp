#include "passes/semantics_inferer.h"
#include "passes/semantics_inferer_pass/utils.h"
#include <iostream>
#include "io/console_graph_exporter.h"

using namespace graph_engine;

namespace passes {

	auto SemanticsInferer::transform_graph(Graph& graph) -> void {
		for (NodeID node_id = 0; node_id < graph.nodes.size(); ++node_id) {
			passes::semantics_inferer::transform_node(graph, node_id);
		}
	}
	
}