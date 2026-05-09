#pragma once
#include <unordered_map>
#include <vector>

#include "passes/mlir_conversion_pass/utils.h"
#include "passes/mlir_conversion_pass/mlir_conversion_pass.h"
#include "graph/graph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace graph_engine;
using LinalgRegionBuilder = std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;



namespace passes::mlir_conversion {

    auto tranform_graph(mlir::MLIRContext& context_, const graph_engine::Graph& graph_) -> mlir::OwningOpRef<mlir::ModuleOp> {
        MLIRConversionPass conversion_pass{graph_, context_ };
        mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
        return model;
    }

    auto datatype_to_mlir_type(mlir::OpBuilder& builder, const graph_engine::DataType dtype) -> mlir::Type {
        mlir::Type return_type;
        switch (dtype) {
        case DataType::BOOL:
            return_type = builder.getI1Type();
            break;
        case DataType::FLOAT32:
            return_type = builder.getF32Type();
            break;
        case DataType::INT64:
            return_type = builder.getI64Type();
            break;
        default:
            throw std::runtime_error("Conversion to MLIR: Invalid graph_engine::DataType encountered");
        }
        return return_type;
    }

    auto get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, graph_engine::ValueID value_id) -> mlir::RankedTensorType {
        mlir::Type dtype = datatype_to_mlir_type(builder, graph.values[value_id].dtype);
        const Shape& s = graph.values[value_id].shape;
        llvm::SmallVector<int64_t, graph_engine::MAX_VALUE_RANK> shape_slice(s.begin(), s.end());
        return mlir::RankedTensorType::get(shape_slice, dtype);
    }

    auto get_function_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph) -> mlir::FunctionType {
        std::vector<mlir::Type> inputs;
        std::vector<mlir::Type> outputs;

        inputs.reserve(graph.inputs.size());
        outputs.reserve(graph.outputs.size());

        std::transform(graph.inputs.begin(), graph.inputs.end(), std::back_inserter(inputs),
            [&](ValueID v) { return get_value_tensor_type(builder, graph, v); });
        std::transform(graph.outputs.begin(), graph.outputs.end(), std::back_inserter(outputs),
            [&](ValueID v) { return get_value_tensor_type(builder, graph, v); });

        return builder.getFunctionType(inputs, outputs);
    }

    auto matmul(mlir::Value a, mlir::Value b, mlir::OpBuilder& builder, mlir::Location loc, bool transpose_b) -> mlir::Value {

        int64_t M = a.getType().cast<mlir::RankedTensorType>().getDimSize(0);       // A is (M x K)
        unsigned short N_index = (unsigned short)(!transpose_b);
        int64_t N = b.getType().cast<mlir::RankedTensorType>().getDimSize(N_index); // B is (K x N)

        mlir::Type element_type = a.getType().cast<mlir::RankedTensorType>().getElementType();

        // Allocate tensor for matmul typing : RankedTensor (M x N)
        mlir::Value matmul_alloc = builder.create<mlir::tensor::EmptyOp>(
            loc,
            mlir::ArrayRef<int64_t>{M, N},
            element_type
        );

        // Fill allocated tensor with zeros, as linalg::matmul performs in_place op: C = C + A @ B
        mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getFloatAttr(element_type, 0.0));

        mlir::Value matmul_init = builder.create<mlir::linalg::FillOp>(loc, zero, matmul_alloc).getResult(0);

        mlir::Operation* matmul_op;
        if (transpose_b) {
            matmul_op = builder.create<mlir::linalg::MatmulTransposeBOp>(
                loc,
                mlir::TypeRange{ matmul_init.getType() },   //result TensorType
                mlir::ValueRange{ a, b },                   //inputs
                mlir::ValueRange{ matmul_init }             //outputs
            );
        }
        else {
            matmul_op = builder.create<mlir::linalg::MatmulOp>(
                loc,
                mlir::TypeRange{ matmul_init.getType() },   //result TensorType
                mlir::ValueRange{ a, b },                   //inputs
                mlir::ValueRange{ matmul_init }             //outputs
            );
        };

        // because SSA, matmul_init is a start state of output of operation (zeroed tensor), 
        // and is not a result of matmul
        mlir::Value matmul_result = matmul_op->getResult(0);
        return matmul_result;
    };

    auto scalar_mul(mlir::Value A, float s, mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Value {
        auto shaped_type = A.getType().cast<mlir::ShapedType>();
        auto element_type = shaped_type.getElementType();

        mlir::Value scalar = builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(s), element_type.cast<mlir::FloatType>());
        mlir::Value init = builder.create<mlir::tensor::EmptyOp>(loc, shaped_type.getShape(), element_type);

        // scalar map should take same input dims as tensor - mlir::Value A;
        auto rank = shaped_type.getRank();
        auto scalar_map = mlir::AffineMap::get(rank, 0, {}, builder.getContext());

        // linalg::generic for scalar mul:
        llvm::SmallVector<mlir::AffineMap> indexing_maps = {
            scalar_map,                                             // scalar
            builder.getMultiDimIdentityMap(shaped_type.getRank()),  // Tensor A
            builder.getMultiDimIdentityMap(shaped_type.getRank())   // Tensor output
        };

        llvm::SmallVector<mlir::utils::IteratorType> iter_types(
            shaped_type.getRank(), mlir::utils::IteratorType::parallel);
        //auto iterTypesAttr = builder.getArrayAttr(iterTypes);

        auto generic_op = builder.create<mlir::linalg::GenericOp>(
            loc,
            shaped_type,
            mlir::ValueRange{ scalar, A }, // inputs
            mlir::ValueRange{ init },      // outputs
            indexing_maps,
            iter_types,
            [&](mlir::OpBuilder& nestedBuilder, mlir::Location nestedLoc, mlir::ValueRange args) {
                // args[0] = , args[1] = элемент A
                mlir::Value mul = nestedBuilder.create<mlir::arith::MulFOp>(
                    nestedLoc,
                    args[0],  // scalar
                    args[1]   // A[i,j] - element
                );
                nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc, mul);
            });

        return generic_op.getResult(0);
    }
};