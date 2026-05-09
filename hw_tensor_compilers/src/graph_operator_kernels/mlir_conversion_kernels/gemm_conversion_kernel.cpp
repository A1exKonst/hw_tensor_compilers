#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/utils.h"
#include "graph_operator_kernels/mlir_conversion_kernels/gemm_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto GemmConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        // Operation: result = alpha*A @ B + beta*C;

        mlir::Value input_A = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        mlir::Value input_B = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);
        mlir::Value input_C = storage.convert_graph_value(graph.nodes[producer_node].inputs[2]);

        bool transB = bool(std::get<int64_t>(graph.nodes[producer_node].attr.at("transB")));
        float alpha = std::get<float>(graph.nodes[producer_node].attr.at("alpha"));
        float beta = std::get<float>(graph.nodes[producer_node].attr.at("beta"));

        mlir::Value matmul_result = mlir_conversion::matmul(input_A, input_B, builder, loc, transB);
        mlir::Value alpha_result = mlir_conversion::scalar_mul(matmul_result, alpha, builder, loc);
        mlir::Value beta_result = mlir_conversion::scalar_mul(input_C, beta, builder, loc);


        // elementwise addition : linalg.generic with broadcasting:
        auto result_type = alpha_result.getType().cast<mlir::RankedTensorType>();
        auto bias_type = beta_result.getType().cast<mlir::RankedTensorType>();

        // Indexation maps:
        // (d0, d1) -> (d0, d1) - matmul_result
        // (d0, d1) -> (d1)     - input_C
        auto matrix_map = builder.getMultiDimIdentityMap(2);
        auto vector_map = mlir::AffineMap::get(2, 0, { builder.getAffineDimExpr(1) }, builder.getContext());

        llvm::SmallVector<mlir::AffineMap, 3> indexing_maps = { matrix_map, vector_map, matrix_map };
        llvm::SmallVector<mlir::utils::IteratorType, 2> iterators(2, mlir::utils::IteratorType::parallel);

        result = builder.create<mlir::linalg::GenericOp>(
            loc,
            result_type,
            mlir::ValueRange{ alpha_result, beta_result },
            alpha_result, // выходной тензор для инициализации (out-of-place)
            indexing_maps,
            iterators,
            [&](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
                auto sum = b.create<mlir::arith::AddFOp>(loc, args[0], args[1]);
                b.create<mlir::linalg::YieldOp>(loc, mlir::ValueRange{ sum });
            }
        ).getResult(0);

        return result;
	}
}