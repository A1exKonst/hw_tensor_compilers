#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "graph_operator_kernels/mlir_conversion_kernels/relu_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;
using LinalgRegionBuilder = std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;



namespace passes::mlir_conversion {

	auto ReluConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;
        mlir::Value input = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        //mlir::Value input = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[0]);
        mlir::Type elementType = mlir::cast<mlir::ShapedType>(input.getType()).getElementType();

        // ==== create zero tensor (for comparison in relu):

        mlir::TypedAttr zero_attr;
        if (elementType.isa<mlir::FloatType>()) {
            zero_attr = builder.getFloatAttr(elementType, 0.0);
        }
        else {
            zero_attr = builder.getIntegerAttr(elementType, 0);
        }
        mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, zero_attr);

        mlir::Value output = builder.create<mlir::tensor::EmptyOp>(
            loc, mlir::cast<mlir::RankedTensorType>(input.getType()).getShape(), elementType);

        // ==== create elementwise operation map:
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        mlir::AffineMap map = builder.getMultiDimIdentityMap(rank); // elementwise operation -> maps are identical
        llvm::SmallVector<mlir::AffineMap> maps = { map, map };     // 2 maps : for input + output

        // ==== create elementwise iterators:
        // an iterator is required for each dim, {rank} iterators total
        // element operations are independent -> iterators are parallel
        llvm::SmallVector<mlir::utils::IteratorType> iterators(rank, mlir::utils::IteratorType::parallel);

        // ==== choose arith::Max operation:
        LinalgRegionBuilder lambda_arith_max;
        if (elementType.isa<mlir::FloatType>()) {
            lambda_arith_max = [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange args) {
                auto max = b.create<mlir::arith::MaximumFOp>(l, args[0], zero);  // args[0] - input element
                b.create<mlir::linalg::YieldOp>(l, max.getResult());             // args[1] - output element
                };
        }
        else {
            lambda_arith_max = [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange args) {
                auto max = b.create<mlir::arith::MaxSIOp>(l, args[0], zero);
                b.create<mlir::linalg::YieldOp>(l, max.getResult());
                };
        }

        // ==== create RELU (linalg.generic)
        mlir::linalg::GenericOp relu_op;
        relu_op = builder.create<mlir::linalg::GenericOp>(
            loc,                // location 
            input.getType(),    // result type
            input,              // relu input
            output,             // relu output
            maps,               // elementwise operation maps
            iterators,          // elementwise operation iterators
            lambda_arith_max);  // operation

        result = relu_op.getResult(0);

        return result;
	}
}