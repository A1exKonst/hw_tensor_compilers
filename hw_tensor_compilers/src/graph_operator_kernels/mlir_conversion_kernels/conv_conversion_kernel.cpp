#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/utils.h"
#include "graph_operator_kernels/mlir_conversion_kernels/conv_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto ConvConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        // Given : mlir::Value input; mlir::Value filter
        // Dialects : linalg, arith, tensor
        // 
        // Expected output:
        // mlir::Value result = conv_op.getResult(0)

        mlir::Value input = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        mlir::Value filter = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);

        auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
        auto filter_type = mlir::cast<mlir::RankedTensorType>(filter.getType());
        auto element_type = input_type.getElementType();
        auto output_type = mlir_conversion::get_value_tensor_type(builder, graph, value_id);

        mlir::Value init_tensor = builder.create<mlir::tensor::EmptyOp>(loc, output_type.getShape(), output_type.getElementType());
        mlir::Value dest;

        // init destination with zeroes:
        if (graph.nodes[producer_node].inputs.size() < 3) {
            mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(element_type));
            dest = builder.create<mlir::linalg::FillOp>(loc, zero, init_tensor).result();
        }
        else { // V3 as bias arg in Convolution:
            mlir::Value bias = storage.convert_graph_value(graph.nodes[producer_node].inputs[2]);
            auto bias_type = mlir::cast<mlir::RankedTensorType>(bias.getType());
            int64_t last_dim = output_type.getRank() - 1;

            // fill init_tensor with bias for all H, W
            dest = builder.create<mlir::linalg::BroadcastOp>(
                loc,
                bias,
                init_tensor,
                mlir::ArrayRef<int64_t>{0, 2, 3} // broadcasted dims
            ).getResults()[0];
        }

        // Affine Maps :
        // Iterators and Conv indices: d0=N, d1=H, d2=W, d3=F, d4=KH, d5=KW, d6=C
        auto map_input = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(0),
                                                     builder.getAffineDimExpr(1) + builder.getAffineDimExpr(4),
                                                     builder.getAffineDimExpr(2) + builder.getAffineDimExpr(5),
                                                     builder.getAffineDimExpr(6) }, builder.getContext());
        auto map_filter = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(4),
                                                      builder.getAffineDimExpr(5),
                                                      builder.getAffineDimExpr(6),
                                                      builder.getAffineDimExpr(3) }, builder.getContext());
        auto map_output = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(0),
                                                      builder.getAffineDimExpr(1),
                                                      builder.getAffineDimExpr(2),
                                                      builder.getAffineDimExpr(3) }, builder.getContext());

        // Iterators:
        llvm::SmallVector<mlir::AffineMap, 3> indexing_maps = { map_input, map_filter, map_output };
        llvm::SmallVector<mlir::utils::IteratorType> iterTypes(4, mlir::utils::IteratorType::parallel);
        iterTypes.append(3, mlir::utils::IteratorType::reduction);
        auto convOp = builder.create<mlir::linalg::GenericOp>(
            loc,
            output_type,
            mlir::ValueRange{ input, filter }, // V0, V2
            mlir::ValueRange{ dest },          // V1 (output buffer)
            indexing_maps,
            iterTypes,
            [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
                // args[0] - input, args[1] - filter, args[2] - output_acc
                mlir::Value mul = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                mlir::Value add = builder.create<mlir::arith::AddFOp>(loc, mul, args[2]);
                builder.create<mlir::linalg::YieldOp>(loc, add);
            });
        result = convOp.getResult(0);

        return result;
	}
}