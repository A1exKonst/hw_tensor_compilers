#include <unordered_map>
#include <memory>
#include <cassert>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "graph_operator_kernels/mlir_conversion_kernels/matmul_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;
using LinalgRegionBuilder = std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;



namespace passes::mlir_conversion {

	auto MatmulConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;
        mlir::Value a = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        mlir::Value b = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);

        auto a_type = llvm::cast<mlir::RankedTensorType>(a.getType());
        auto b_type = llvm::cast<mlir::RankedTensorType>(b.getType());
        auto element_type = a_type.getElementType();
        int64_t rank = a_type.getRank();

        assert(a_type.getElementType() == b_type.getElementType());
        assert(a_type.getRank() == b_type.getRank());
        assert(a_type.getRank() > 1);
        assert(a_type.getShape()[rank - 1] == b_type.getShape()[rank - 2]);

        mlir::Value zero_attr = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(element_type));

        if (rank == 2) {
            int64_t m = a_type.getDimSize(0);
            int64_t n = b_type.getDimSize(1);
            auto result_type = mlir::RankedTensorType::get({ m, n }, element_type);
            mlir::Value empty_tensor = builder.create<mlir::tensor::EmptyOp>(loc, result_type.getShape(), element_type);
            mlir::Value init_tensor = builder.create<mlir::linalg::FillOp>(loc, zero_attr, empty_tensor).getResult(0);
            auto matmul_op = builder.create<mlir::linalg::MatmulOp>(
                loc,
                result_type,
                mlir::ValueRange{ a, b },       // inputs
                mlir::ValueRange{ init_tensor } // outputs
            );
            return matmul_op.getResult(0);
        }

        if (rank == 3) {
            auto a_shape = a_type.getShape();
            auto b_shape = b_type.getShape();

            int64_t batch_size = a_shape[0];
            int64_t M = a_shape[rank - 2];
            int64_t K = a_shape[rank - 1];
            int64_t N = b_shape[rank - 1];

            // result init tensor
            auto result_type = mlir::RankedTensorType::get({ batch_size, M, N }, element_type);
            mlir::Value empty_tensor = builder.create<mlir::tensor::EmptyOp>(loc, result_type.getShape(), element_type);
            mlir::Value init_tensor = builder.create<mlir::linalg::FillOp>(loc, zero_attr, empty_tensor).getResult(0);

            // matmul op
            auto matmul_op = builder.create<mlir::linalg::BatchMatmulOp>(
                loc,
                result_type,
                mlir::ValueRange{ a, b },
                mlir::ValueRange{ init_tensor }
            );
            result = matmul_op.getResult(0);
            return result;
        }

        auto a_shape = a_type.getShape();
        auto b_shape = b_type.getShape();

        // calc batch_size
        int64_t batch_size = 1;
        for (int64_t i = 0; i < rank - 2; ++i) {
            batch_size *= a_shape[i];
        }

        int64_t M = a_shape[rank - 2];
        int64_t K = a_shape[rank - 1];
        int64_t N = b_shape[rank - 1];

        // reassociation indices for batch tensor::collapse_shape
        llvm::SmallVector<mlir::ReassociationIndices, 4> reassociation;
        mlir::ReassociationIndices batch_indices;
        for (int64_t i = 0; i < rank - 2; ++i) {
            batch_indices.push_back(i);
        }
        reassociation.push_back(batch_indices); // Group 0: batch indices
        reassociation.push_back({ rank - 2 });  // Group 1: M or K
        reassociation.push_back({ rank - 1 });  // Group 2: K or N

        auto a_type_collapsed = mlir::RankedTensorType::get({ batch_size, M, K }, element_type);
        auto b_type_collapsed = mlir::RankedTensorType::get({ batch_size, K, N }, element_type);

        mlir::Value a_collapsed = builder.create<mlir::tensor::CollapseShapeOp>(loc, a_type_collapsed, a, reassociation);
        mlir::Value b_collapsed = builder.create<mlir::tensor::CollapseShapeOp>(loc, b_type_collapsed, b, reassociation);

        // result init tensor
        auto result_type_collapsed = mlir::RankedTensorType::get({ batch_size, M, N }, element_type);
        mlir::Value empty_tensor = builder.create<mlir::tensor::EmptyOp>(loc, result_type_collapsed.getShape(), element_type);
        mlir::Value init_tensor = builder.create<mlir::linalg::FillOp>(loc, zero_attr, empty_tensor).getResult(0);

        // matmul op
        auto matmul_op = builder.create<mlir::linalg::BatchMatmulOp>(
            loc, 
            result_type_collapsed, 
            mlir::ValueRange{ a_collapsed, b_collapsed}, 
            mlir::ValueRange{ init_tensor }
        );
        mlir::Value result_collapsed = matmul_op.getResult(0);

        // expand shape
        llvm::SmallVector<int64_t, 4> result_shape;
        for (int64_t i = 0; i < rank - 2; ++i) {
            result_shape.push_back(a_shape[i]);
        }
        result_shape.push_back(M);
        result_shape.push_back(N);
        auto result_type = mlir::RankedTensorType::get(result_shape, element_type);

        result = builder.create<mlir::tensor::ExpandShapeOp>(loc, result_type, result_collapsed, reassociation);
        return result;
	}

}