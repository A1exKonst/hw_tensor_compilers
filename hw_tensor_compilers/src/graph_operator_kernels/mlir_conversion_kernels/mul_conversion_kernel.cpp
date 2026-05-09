#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "graph_operator_kernels/mlir_conversion_kernels/mul_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto MulConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        // result = create_binary_operation<mlir::arith::MulIOp, mlir::arith::MulFOp>(producer_node);
        mlir::Value lhs = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        mlir::Value rhs = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);
        //mlir::Value lhs = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[0]);
        //mlir::Value rhs = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[1]);
        auto tensor_type = lhs.getType().cast<mlir::RankedTensorType>();
        auto shape = tensor_type.getShape();
        int64_t rank = tensor_type.getRank();

        // empty result tensor
        mlir::Value init_tensor = builder.create<mlir::tensor::EmptyOp>(loc, shape, tensor_type.getElementType());

        // maps : elementwise operation : (d0, d1, ...) -> (d0, d1, ...)
        auto map = builder.getMultiDimIdentityMap(rank);
        llvm::SmallVector<mlir::AffineMap, 3> indexing_maps(3, map);

        // iterators : elementwise operation : all IteratorType::parallel
        llvm::SmallVector<mlir::utils::IteratorType, 2> iter_types(rank, mlir::utils::IteratorType::parallel);

        // linalg.generic :
        auto elementwise_op = builder.create<mlir::linalg::GenericOp>(
            loc,
            /*resultTypes=*/tensor_type,
            /*inputs=*/mlir::ValueRange{ lhs, rhs },
            /*outputs=*/mlir::ValueRange{ init_tensor },
            indexing_maps,
            iter_types,
            [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
                // args[0] = lhs_scalar, args[1] = rhs_scalar, args[2] = out_scalar
                mlir::Value mul;
                if (tensor_type.getElementType().isa<mlir::FloatType>()) {
                    mul = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                }
                else {
                    mul = builder.create<mlir::arith::MulIOp>(loc, args[0], args[1]);
                }
                builder.create<mlir::linalg::YieldOp>(loc, mul);
            }
        );

        result = elementwise_op.getResult(0);

        return result;
	}
}