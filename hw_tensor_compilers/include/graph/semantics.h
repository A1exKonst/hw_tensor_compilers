#pragma once
#include "graph.h"

namespace semantics {
	void decorate_graph(graph_engine::Graph& graph);

	void decorate_graph(graph_engine::Graph& graph, graph_engine::NodeID node_id);

	void expect_dtype(graph_engine::Graph& graph, 
					  graph_engine::ValueID value_id, 
					  graph_engine::DataType dtype);

	void expect_shape(graph_engine::Graph& graph, 
					  graph_engine::ValueID value_id, 
					  graph_engine::Shape shape);

	graph_engine::DataType math_result_data_type(graph_engine::DataType dt1, graph_engine::DataType dt2);
};