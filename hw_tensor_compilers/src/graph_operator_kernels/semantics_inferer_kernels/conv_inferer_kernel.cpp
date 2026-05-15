#pragma once
#include "graph_operator_kernels/semantics_inferer_kernels/conv_inferer_kernel.h"
#include "graph/graph.h"
#include "passes/semantics_inferer_pass/utils.h"

using namespace graph_engine;



namespace passes::semantics_inferer {

	auto ConvInfererKernel::transform_node(graph_engine::Graph& graph, const graph_engine::NodeID node_id) -> void {
		Node& node = graph.nodes.at(node_id);
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
		}
		expect(graph.values[x_id].shape[1] / group == graph.values[w_id].shape[1], "Node[Conv] : W[1] != C / group");
		if (is_b_initialized) {
			expect(graph.values[b_id].shape.rank() == 1, "Node[Conv] : B.rank != 1");
			expect(graph.values[b_id].shape[0] == graph.values[w_id].shape[0], "Node[Conv] : B.shape[0] != M");
		}

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
		return;
	}

}