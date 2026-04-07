#pragma once
#include "graph/graph.h"

namespace passes
{
	class SemanticsInferer {
	public:
		SemanticsInferer() {};
		SemanticsInferer(const SemanticsInferer& other) = default;
		SemanticsInferer(SemanticsInferer&& other) = default;

		~SemanticsInferer() = default;

		SemanticsInferer& operator=(const SemanticsInferer&) = default;
		SemanticsInferer& operator=(SemanticsInferer&&) = default;

		static void transform_graph(graph_engine::Graph& graph);

	private:
		static void transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node);

		static void expect_dtype(graph_engine::Graph& graph,
			const graph_engine::ValueID value_id,
			const graph_engine::DataType dtype);

		static void expect_shape(graph_engine::Graph& graph,
			const graph_engine::ValueID value_id,
			const graph_engine::Shape shape);

		static void expect_attribute(graph_engine::Graph& graph, 
			graph_engine::NodeID node_id, 
			const std::string& name, 
			graph_engine::AttributeValue attr);

		static void expect(bool assertion, std::string&& error_message);

		static void insert_type_conversion(graph_engine::Graph& graph, 
			graph_engine::ValueID converted_value_id, 
			graph_engine::DataType new_dtype);
	};
}