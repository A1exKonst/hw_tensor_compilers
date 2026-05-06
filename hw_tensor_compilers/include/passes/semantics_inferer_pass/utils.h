#pragma once
#include "graph/graph.h"

namespace passes::semantics_inferer {

	auto transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void;

	auto expect_dtype(graph_engine::Graph& graph, const graph_engine::ValueID value_id, const graph_engine::DataType dtype) -> void;

	auto expect_shape(graph_engine::Graph& graph, const graph_engine::ValueID value_id, graph_engine::Shape shape) -> void;

	auto expect(bool assertion, std::string&& error_message) -> void;

	auto expect_attribute(graph_engine::Graph& graph, graph_engine::NodeID node_id, const std::string& name, graph_engine::AttributeValue attr) -> void;

	auto insert_type_conversion(graph_engine::Graph& graph, graph_engine::ValueID converted_value_id, graph_engine::DataType new_dtype) -> void;

}