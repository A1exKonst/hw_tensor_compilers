#include <memory>

#include "passes/semantics_inferer_pass/semantics_inferer_pass.h"
#include "graph/node.h"
#include "graph_operator_kernels/semantics_inferer_kernels/add_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/constant_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/conv_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/elementwise_binop_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/gemm_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/matmul_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/mul_inferer_kernel.h"
#include "graph_operator_kernels/semantics_inferer_kernels/relu_inferer_kernel.h"

using namespace graph_engine;



namespace passes {

	SemanticsInfererPass::SemanticsInfererPass() {
		// fill registry_
		registry_[OperatorType::ADD] = std::make_unique<semantics_inferer::AddInfererKernel>();
		registry_[OperatorType::CONSTANT] = std::make_unique<semantics_inferer::ConstantInfererKernel>();
		registry_[OperatorType::CONV] = std::make_unique<semantics_inferer::ConvInfererKernel>();
		registry_[OperatorType::GEMM] = std::make_unique<semantics_inferer::GemmInfererKernel>();
		registry_[OperatorType::MATMUL] = std::make_unique<semantics_inferer::MatmulInfererKernel>();
		registry_[OperatorType::MUL] = std::make_unique<semantics_inferer::MulInfererKernel>();
		registry_[OperatorType::RELU] = std::make_unique<semantics_inferer::ReluInfererKernel>();

	}

	auto SemanticsInfererPass::transform_graph(graph_engine::Graph& graph) -> void {
		for (NodeID node_id = 0; node_id < graph.nodes.size(); ++node_id) {
			registry_.at(graph.nodes[node_id].op_type)->transform_node(graph, node_id);
		}
	}

}