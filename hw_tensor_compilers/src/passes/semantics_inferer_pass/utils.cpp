#include "passes/semantics_inferer_pass/utils.h"
#include <iostream>
#include "io/console_graph_exporter.h"

using namespace graph_engine;

namespace passes::semantics_inferer {

	auto transform_node(Graph& graph, const NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);
		switch (node.op_type) {
		case OperatorType::ADD:
		case OperatorType::MUL: {
			expect(node.inputs.size() == 2, "Node[Add || Mul] : two input Values are expected");
			expect(node.outputs.size() == 1, "Node[Add || Mul] : one output Value is expected");

			ValueID out = node.outputs.at(0);
			ValueID first = node.inputs.at(0);
			ValueID second = node.inputs.at(1);

			DataType result_type = graph_engine::math_result_data_type(
				graph.values.at(first).dtype,
				graph.values.at(second).dtype);
			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, out, result_type);

			std::optional<Shape> result_shape = graph_engine::calculate_broadcast_compatible_shape(
				graph.values[first].shape,
				graph.values[second].shape);
			expect(result_shape.has_value(), "Node[Add || Mul] : couldn't broadcast shapes");
			expect_shape(graph, first, result_shape.value());
			expect_shape(graph, second, result_shape.value());
			expect_shape(graph, out, std::move(result_shape.value()));
			break;
		}
		case OperatorType::CONSTANT: {
			expect(node.outputs.size() == 1, "Node[Constant] : one output Value is expected");
			break;
		}
		case OperatorType::CONV: {
			using AttrSeq = std::vector<int64_t>;

			size_t inputs_size = node.inputs.size();
			expect((inputs_size == 2 || inputs_size == 3), "Node[Conv] : 2 or 3 input Values are expected");
			expect(node.outputs.size() == 1, "Node[Conv] : one output Value is expected");

			ValueID x_id = node.inputs[0];
			ValueID w_id = node.inputs[1];
			ValueID y_id = node.outputs[0];
			ValueID b_id = ValueID(-1);
			bool is_b_initialized = (inputs_size > 2);
			if (is_b_initialized) b_id = node.inputs[2];

			// =========================== Datatype Inference ==================================================

			DataType result_type = math_result_data_type(graph.values[x_id].dtype, graph.values[w_id].dtype);
			if (is_b_initialized) result_type = math_result_data_type(result_type, graph.values[b_id].dtype);

			expect_dtype(graph, x_id, result_type);
			expect_dtype(graph, w_id, result_type);
			expect_dtype(graph, y_id, result_type);
			if (is_b_initialized) expect_dtype(graph, b_id, result_type);


			// =========================== Shape Inference =====================================================

			Shape y_shape(graph.values[x_id].shape.rank());
			expect(graph.values[x_id].shape.rank() >= 3, "Node[Conv] : X.rank() < 3");
			expect(graph.values[x_id].shape.rank() == graph.values[w_id].shape.rank(), "Node[Conv] : rank(X) != rank(W)");

			auto rank = graph.values[x_id].shape.rank();
			auto spatial_rank = rank - 2;
			auto N = graph.values[x_id].shape[0];
			auto C = graph.values[x_id].shape[1];
			auto M = graph.values[w_id].shape[0];

			// set default attributes:
			int64_t group = 1;
			AttrSeq pads(spatial_rank * 2, 2);
			AttrSeq kernel_shape(graph.values[w_id].shape.begin() + 2, graph.values[w_id].shape.end());
			AttrSeq strides(spatial_rank, 1);
			AttrSeq dilations(spatial_rank, 1);
			std::string auto_pad = "NOTSET";

			// read attributes:
			if (node.attr.find("group") != node.attr.end()) group = std::get<int64_t>(node.attr.at("group"));
			if (node.attr.find("strides") != node.attr.end()) strides = std::get<AttrSeq>(node.attr.at("strides"));
			if (node.attr.find("dilations") != node.attr.end()) dilations = std::get<AttrSeq>(node.attr.at("dilations"));
			if (node.attr.find("kernel_shape") != node.attr.end()) kernel_shape = std::get<AttrSeq>(node.attr.at("kernel_shape"));
			if (node.attr.find("pads") != node.attr.end())  pads = std::get<AttrSeq>(node.attr.at("pads"));
			if (node.attr.find("auto_pad") != node.attr.end()) auto_pad = std::get<std::string>(node.attr.at("auto_pad"));

			// expect attributes validity:
			expect((group > 0 && C % group == 0 && M % group == 0), "Node[Conv] : invalid \"group\" value");
			expect(strides.size() == spatial_rank, "Node[Conv] : strides.size() != spatial_rank");
			for (int64_t s : strides) expect(s > 0, "Node[Conv] : strides[i] <= 0");
			expect(dilations.size() == spatial_rank, "Node[Conv] : dilations.size() != spatial_rank");
			for (int64_t d : dilations) expect(d >= 1, "Node[Conv] : dilations[i] < 1");
			expect((kernel_shape.size() == spatial_rank), "Node[Conv] : kernel_shape.size() != spatial_rank");
			for (int i = 0; i < kernel_shape.size(); ++i) {
				expect(graph.values[w_id].shape[i + 2] == kernel_shape[i], "Node[Conv] : kernel_shape != W[2:]");
			};
			expect(graph.values[x_id].shape[1] / group == graph.values[w_id].shape[1], "Node[Conv] : W[1] != C / group");
			if (is_b_initialized) {
				expect(graph.values[b_id].shape.rank() == 1, "Node[Conv] : B.rank != 1");
				expect(graph.values[b_id].shape[0] == graph.values[w_id].shape[0], "Node[Conv] : B.shape[0] != M");
			};

			// form Y shape:
			y_shape[0] = N;
			y_shape[1] = M;
			for (int i = 0; i < spatial_rank; ++i) {
				auto in_dim = graph.values[x_id].shape[i + 2];
				auto k_dim = kernel_shape[i];
				auto d_kernel = dilations[i] * (kernel_shape[i] - 1) + 1;

				int64_t out_dim = 0;
				if (auto_pad == "NOTSET") {
					auto p_begin = pads[i];
					auto p_end = pads[i + spatial_rank];
					out_dim = (in_dim + p_begin + p_end - d_kernel) / strides[i] + 1;
				}
				else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
					out_dim = in_dim / strides[i];
				}
				else if (auto_pad == "VALID") {
					out_dim = (in_dim - d_kernel + 1) / strides[i];
				}
				else throw std::runtime_error("Node[Conv] : invalid auto_pad value <" + auto_pad + ">");
				y_shape[i + 2] = out_dim;
			}

			expect_shape(graph, y_id, y_shape);
			break;
		}
		case OperatorType::GEMM: {
			// expect input output arguments amount:
			expect(node.outputs.size() == 1, "Node[Gemm] : one output Value is expected");
			expect(node.inputs.size() == 3, "Node[Gemm] : 3 input Values are expected");


			// expect dtypes:
			ValueID out = node.outputs.at(0);
			ValueID first = node.inputs.at(0);
			ValueID second = node.inputs.at(1);
			ValueID third = node.inputs.at(2);

			Value& first_val = graph.values.at(first);
			Value& second_val = graph.values.at(second);
			Value& third_val = graph.values.at(third);

			DataType result_type = graph_engine::math_result_data_type(
				graph_engine::math_result_data_type(
					first_val.dtype,
					second_val.dtype),
				third_val.dtype);

			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, third, result_type);
			expect_dtype(graph, out, result_type);


			// expect shapes:
			expect(first_val.shape.rank() == 2 && second_val.shape.rank() == 2,
				"Values for Gemm : only rank == 2 allowed");

			first_val.shape.rank(2);
			second_val.shape.rank(2);

			int64_t is_transposed_B = std::get<int64_t>(graph.nodes[node_id].attr.at("transB"));
			std::optional<Shape> matmul_shape;
			if (!is_transposed_B) {
				matmul_shape = graph_engine::calculate_matmul_compatible_shape(first_val.shape, second_val.shape);
			}
			else {
				matmul_shape = graph_engine::calculate_matmul_compatible_shape(first_val.shape, transposed(second_val.shape));
			}
			expect(matmul_shape.has_value(), "Values for Gemm : cannot multiply matrices");

			std::optional<Shape> gemm_shape = graph_engine::calculate_broadcast_compatible_shape(matmul_shape.value(), third_val.shape);
			expect(gemm_shape.has_value(), "Values for Gemm : cannot add matrices");
			expect_shape(graph, out, gemm_shape.value());
			/*
			unsigned short M = first_val.shape[0];
			unsigned short N = (first_val.shape[1] > second_val.shape[0]) ? first_val.shape[1] : second_val.shape[0];
			unsigned short K = second_val.shape[1];

			expect(first_val.shape[1] == 0 || first_val.shape[1] == N,
				"Values for Gemm : cannot multiply matrices");
			expect(second_val.shape[0] == 0 || second_val.shape[0] == N,
				"Values for Gemm : cannot multiply matrices");
			first_val.shape[1] = N;
			second_val.shape[0] = N;
			expect(
				((third_val.shape[0] == 0 || third_val.shape[0] == M) &&
					(third_val.shape[1] == 0 || third_val.shape[1] == K)) ||
				((third_val.shape[1] == 0 || third_val.shape[1] == M) &&
					(third_val.shape[0] == 0 || third_val.shape[0] == K)),
				"Values for Gemm : cannot add matrices");
			third_val.shape[0] = M;
			third_val.shape[1] = K;
			expect_shape(graph, out, third_val.shape);
			*/
			break;
		}
		case OperatorType::INPUT: {
			expect(false, "Node[Input] : not supported");
			break;
		}
		case OperatorType::MATMUL: {
			expect(node.outputs.size() == 1,
				"Node[Gemm] : one output Value is expected");
			expect(node.inputs.size() == 2,
				"Node[Gemm] : 3 input Values are expected");

			ValueID out = node.outputs.at(0);
			ValueID first = node.inputs.at(0);
			ValueID second = node.inputs.at(1);

			// expect dtypes:
			DataType result_type = graph_engine::math_result_data_type(graph.values.at(first).dtype, graph.values.at(second).dtype);
			expect_dtype(graph, first, result_type);
			expect_dtype(graph, second, result_type);
			expect_dtype(graph, out, result_type);

			// expect shapes:
			auto result_rank = graph.values[first].shape.rank();
			expect(result_rank == graph.values[second].shape.rank(),
				"Node[MatMul] : equal ranks of input Values are expected");

			Shape result_shape = Shape(result_rank);

			// expect shapes: Broadcasting:
			for (int i = 0; i < result_rank - 2; ++i) {
				expect(graph.values[first].shape[i] == graph.values[second].shape[i],
					"Node[MatMul] : different broadcasted shape dimensions found");
				result_shape[i] = graph.values[first].shape[i];
			}

			// expect shapes: MatMul last 2 dims:
			expect(graph.values[first].shape[result_rank - 1] == graph.values[second].shape[result_rank - 2],
				"Node[MatMul] : tensors last 2 dims (M*N1 and N2*K). N1 == N2 is expected, but it is false");
			result_shape[result_rank - 2] = graph.values[first].shape[result_rank - 2];  // dimension M
			result_shape[result_rank - 1] = graph.values[second].shape[result_rank - 1]; // dimension K

			graph.values[out].shape = std::move(result_shape);
			break;
		}
		case OperatorType::RELU: {
			expect(node.outputs.size() == 1, "Node[Relu] : one output Value is expected");
			expect(node.inputs.size() == 1, "Node[Relu] : one input Value is expected");
			expect_dtype(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).dtype); // expect equal dtypes
			expect_shape(graph, node.outputs.at(0), graph.values.at(node.inputs.at(0)).shape); // expect equal shapes
			break;
		}
		default: {
			throw std::runtime_error("decorate_graph(graph.nodes[" + std::to_string(node_id) + "]) : no such OperatorType known.");
			break;
		}
		};
	};

	auto expect_dtype(Graph& graph, const ValueID value_id, const DataType dtype) -> void {
		if (graph.values[value_id].dtype == dtype) return;

		if (graph.values[value_id].dtype != DataType::UNDEFINED) {
			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": expected DataType::" + data_type_to_str.at(dtype)
				+ " but DataType::" + data_type_to_str.at(graph.values[value_id].dtype) + " found.");
		};

		graph.values[value_id].dtype = dtype;
		return;
	};

	auto expect_shape(Graph& graph, const ValueID value_id, Shape shape) -> void {
		if (graph.values[value_id].shape == shape) return;

		if (graph.values[value_id].shape.rank() != 0) {
			std::cout << "V" << value_id << ".Shape:        " << graph.values[value_id].shape << std::endl;
			std::cout << "Expected Shape: " << shape << std::endl;

			throw std::runtime_error("V" + std::to_string(value_id)
				+ ": tried to initialize Shape, when it is already initialized");
		};

		graph.values[value_id].shape = std::move(shape);
		return;
	};

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
		};
		node.attr[name] = std::move(attr);
		return;
	};

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
	};

}