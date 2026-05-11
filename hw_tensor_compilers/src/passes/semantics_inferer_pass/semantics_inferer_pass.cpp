#include <memory>

#include "passes/semantics_inferer_pass/semantics_inferer_pass.h"
#include "graph/node.h"
#include "graph_operator_kernels/semantics_inferer_kernels/add_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/constant_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/conv_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/gemm_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/matmul_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/mul_inferer.h"
#include "graph_operator_kernels/semantics_inferer_kernels/relu_inferer.h"

using namespace graph_engine;



namespace passes {

	SemanticsInfererPass::SemanticsInfererPass() {
		// fill registry_
		registry_[OperatorType::ADD] = std::make_unique<semantics_inferer::AddInferer>();
		registry_[OperatorType::CONSTANT] = std::make_unique<semantics_inferer::ConstantInferer>();
		registry_[OperatorType::CONV] = std::make_unique<semantics_inferer::ConvInferer>();
		registry_[OperatorType::GEMM] = std::make_unique<semantics_inferer::GemmInferer>();
		registry_[OperatorType::MATMUL] = std::make_unique<semantics_inferer::MatmulInferer>();
		registry_[OperatorType::MUL] = std::make_unique<semantics_inferer::MulInferer>();
		registry_[OperatorType::RELU] = std::make_unique<semantics_inferer::ReluInferer>();

	}

	auto SemanticsInfererPass::transform_graph(graph_engine::Graph& graph) -> void {
		for (NodeID node_id = 0; node_id < graph.nodes.size(); ++node_id) {
			registry_.at(graph.nodes[node_id].op_type)->transform_node(graph, node_id);
		}
	}

}