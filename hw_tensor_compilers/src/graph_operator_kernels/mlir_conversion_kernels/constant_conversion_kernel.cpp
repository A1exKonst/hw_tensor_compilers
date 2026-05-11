#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/utils.h"
#include "graph_operator_kernels/mlir_conversion_kernels/constant_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto ConstantConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        auto& weights = std::get<std::vector<float>>(graph.nodes[producer_node].attr.at("weights"));
        auto tensor_type = mlir_conversion::get_value_tensor_type(builder, graph, value_id);
        auto weights_attr = mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(weights));
        auto constant_op = builder.create<mlir::arith::ConstantOp>(loc, tensor_type, weights_attr);
        result = constant_op.getResult();

        return result;
	}

}