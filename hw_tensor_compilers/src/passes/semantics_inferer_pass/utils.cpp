#include "passes/semantics_inferer_pass/utils.h"
#include <iostream>
#include "io/console_graph_exporter.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto expect_dtype(Graph& graph, const ValueID value_id, const DataType dtype) -> void {
		if (graph.values[value_id].dtype == dtype) return;

		if (graph.values[value_id].dtype != DataType::UNDEFINED) {
			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": expected DataType::" + data_type_to_str.at(dtype)
				+ " but DataType::" + data_type_to_str.at(graph.values[value_id].dtype) + " found.");
		}

		graph.values[value_id].dtype = dtype;
		return;
	}

	auto expect_shape(Graph& graph, const ValueID value_id, Shape shape) -> void {
		if (graph.values[value_id].shape == shape) return;

		if (graph.values[value_id].shape.rank() != 0) {
			std::cout << "V" << value_id << ".Shape:        " << graph.values[value_id].shape << std::endl;
			std::cout << "Expected Shape: " << shape << std::endl;

			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": tried to initialize Shape, when it is already initialized");
		}

		graph.values[value_id].shape = std::move(shape);
		return;
	}

	auto expect(bool assertion, std::string error_message) -> void {
		if (!assertion) {
			throw std::runtime_error(std::move(error_message));
		}
		return;
	}

	auto expect_attribute(Graph& graph, NodeID node_id, const std::string& name, AttributeValue attr) -> void {
		if (graph.nodes.at(node_id).attr.at(name) == attr) return;

		Node& node = graph.nodes[node_id];
		if (node.attr.find(name) != node.attr.end()) {
			throw std::runtime_error("N" + std::to_string(node_id) +
				": tried to initialize attribute '" + name + "', when it is already initialized");
		}
		node.attr[name] = std::move(attr);
		return;
	}

	auto insert_type_conversion(Graph& graph, ValueID converted_value_id, DataType new_dtype) -> void {
		size_t new_value_expected_id = graph.nodes.size();
		NodeID conversion_node_id = graph.add_node(
			OperatorType::DTYPE_CONVERSION,		// OperatorType
			{ converted_value_id },				// inputs
			{ new_value_expected_id },			// outputs
			{}									// attributes
		);
		ValueID new_value_id = graph.add_value(
			graph.values.at(converted_value_id).shape,	// Shape
			new_dtype,									// DataType
			conversion_node_id							// NodeID producer_id
		);

		graph.nodes[conversion_node_id].inputs.push_back(new_value_id);
		return;
	}

}