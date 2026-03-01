#pragma once
#include "graph/semantics.h"
#include "graph/graph.h"
#include <utility>
#include <type_traits>
#include "io/out_graph_console.h"

using namespace graph_engine;

namespace semantics {

	void expect(bool asserted_value, std::string exception_name);

	void decorate_graph(Graph& graph) {
		for (NodeID node_id = 0; node_id < graph.nodes.size(); ++node_id) {
			decorate_graph(graph, node_id);
		};
	};

	void decorate_graph(Graph& graph, NodeID node_id) {
		Node& node = graph.nodes.at(node_id);
		switch (node.op_type) {
		case OperatorType::ADD:
		case OperatorType::MUL:
		{
			expect(node.outputs.size() == 1, "Node[Add || Mul] : one output Value is expected");

			ValueID out = node.outputs.at(0);
			ValueID first = node.inputs.at(0);
			ValueID second = node.inputs.at(1);

			DataType result_type = math_result_data_type(
				graph.values.at(first).dtype,
				graph.values.at(second).dtype);

			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, out, result_type);

			// expect shapes
			break;
		}
		case OperatorType::CONSTANT:{
			expect(node.outputs.size() == 1, "Node[Constant] : one output Value is expected");
			break;
		}
		case OperatorType::CONV:
			expect(false, "Node[Conv] : not supported");
			break;
		case OperatorType::GEMM:
		{
			// expect input output arguments amount:
			expect(node.outputs.size() == 1,"Node[Gemm] : one output Value is expected");
			expect(node.inputs.size() == 3,	"Node[Gemm] : 3 input Values are expected");


			// expect dtypes:
			ValueID out		= node.outputs.at(0);
			ValueID first	= node.inputs.at(0);
			ValueID second	= node.inputs.at(1);
			ValueID third	= node.inputs.at(2);

			Value& first_val = graph.values.at(first);
			Value& second_val = graph.values.at(second);
			Value& third_val = graph.values.at(third);

			DataType result_type = math_result_data_type(
				math_result_data_type(
					first_val.dtype,
					second_val.dtype),
				third_val.dtype);

			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, third, result_type);
			expect_dtype(graph, out, result_type);

			
			// expect shapes:
			std::cout << "Shapes:" << first_val.shape.rank() << " " << second_val.shape.rank() << std::endl;
			expect(
				(first_val.shape.rank() == 2 || first_val.shape.rank() == 0) &&
				(second_val.shape.rank() == 2 || second_val.shape.rank() == 0),
				"Values for Gemm : only rank == 2 allowed");

			first_val.shape.rank(2);
			second_val.shape.rank(2);
			third_val.shape.rank(2);

			unsigned short M = first_val.shape[0];
			unsigned short N = (first_val.shape[1] > second_val.shape[0]) ? first_val.shape[1] : second_val.shape[0];
			unsigned short K = second_val.shape[1];

			expect(first_val.shape[1] == 0 || first_val.shape[1] == N,
				"Values for Gemm : cannot multiply matrices");
			expect(second_val.shape[0] == 0 || second_val.shape[0] == N,
				"Values for Gemm : cannot multiply matrices");
			first_val.shape[1] = N;
			second_val.shape[0] = N;
			std::cout << third_val.shape << " " << M << " " << K;
			expect(
				((third_val.shape[0] == 0 || third_val.shape[0] == M) &&
				(third_val.shape[1] == 0 || third_val.shape[1] == K)) ||
				((third_val.shape[1] == 0 || third_val.shape[1] == M) &&
				(third_val.shape[0] == 0 || third_val.shape[0] == K)),
				"Values for Gemm : cannot add matrices");
			third_val.shape[0] = M;
			third_val.shape[1] = K;

			expect_shape(graph, out, third_val.shape);
			break;
		}
		case OperatorType::INPUT:{
			expect(false, "Node[Input] : not supported");
			break;
		}
		case OperatorType::MATMUL:
		{
			// expect input output arguments amount:
			expect(node.outputs.size() == 1, "Node[Gemm] : one output Value is expected");
			expect(node.inputs.size() == 2, "Node[Gemm] : 3 input Values are expected");


			// expect dtypes:
			ValueID out = node.outputs.at(0);
			ValueID first = node.inputs.at(0);
			ValueID second = node.inputs.at(1);

			DataType result_type = math_result_data_type(
				graph.values.at(first).dtype,
				graph.values.at(second).dtype);

			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, out, result_type);


			// expect shapes:
			break;
		}
		case OperatorType::RELU:
		{
			expect(node.outputs.size() == 1, "Node[Relu] : one output Value is expected");
			expect(node.inputs.size() == 1,  "Node[Relu] : one input Value is expected");
			expect_dtype(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).dtype); // expect equal dtypes
			expect_shape(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).shape); // expect equal shapes
			break;
		}
		};
	};

	void expect(bool asserted_value, std::string exception_name) {
		if (!asserted_value) {
			throw std::runtime_error(std::move(exception_name));
		};
		return;
	};

	void expect_dtype(Graph& graph, ValueID value_id, DataType dtype) {
		if (graph.values[value_id].dtype == dtype) return;

		if (graph.values[value_id].dtype != DataType::UNDEFINED) {
			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": expected DataType::" + data_type_to_str.at(dtype)
				+ " but DataType::" + data_type_to_str.at(graph.values[value_id].dtype) + " found.");
		};

		graph.values[value_id].dtype = dtype;
		return;
	};

	void insert_type_conversion(Graph& graph, ValueID converted_value_id, DataType new_dtype) {
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
	};

	void expect_shape(Graph& graph, ValueID value_id, Shape shape) {
		if (graph.values[value_id].shape == shape) return;

		if (graph.values[value_id].shape.rank() != 0) {
			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": tried to initialize Shape, when it is already initialized");
		};

		graph.values[value_id].shape = std::move(shape);
		return;
	};

	void expect_attribute(Graph& graph, NodeID node_id, const std::string& name, AttributeValue attr) {
		if (graph.nodes.at(node_id).attr.at(name) == attr) return;

		Node& node = graph.nodes[node_id];
		if (node.attr.contains(name)) {
			throw std::runtime_error("N" + std::to_string(node_id) + 
				": tried to initialize attribute '" + name + "', when it is already initialized");
		};
		node.attr[name] = std::move(attr);
		return;
	};

	DataType math_result_data_type(DataType dt1, DataType dt2) {
		if ((dt1 == DataType::DELETED_VALUE) ||
			(dt2 == DataType::DELETED_VALUE)) {
			throw std::runtime_error("Tried to get math_result.dtype, but it is DataType::DELETED_VALUE");
		};
		if (static_cast<uint8_t>(dt1) < static_cast<uint8_t>(dt2)) return dt2;
		return dt1;
	};

};