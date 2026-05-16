#include <iostream>

#include "passes/mlir_conversion_pass/mlir_conversion_pass.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/mlir_conversion_kernel.h"
#include "passes/mlir_conversion_pass/utils.h"

#include "graph_operator_kernels/mlir_conversion_kernels/add_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/constant_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/conv_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/gemm_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/matmul_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/mul_conversion_kernel.h"
#include "graph_operator_kernels/mlir_conversion_kernels/relu_conversion_kernel.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;



namespace passes {

	MLIRConversionPass::MLIRConversionPass(const graph_engine::Graph& graph__, mlir::MLIRContext& context__) : 
		builder(&context__), graph(graph__), context(context__) {
		// fill registry_
		registry_[OperatorType::ADD] = std::make_unique<mlir_conversion::AddConversionKernel>();
		registry_[OperatorType::CONSTANT] = std::make_unique<mlir_conversion::ConstantConversionKernel>();
		registry_[OperatorType::CONV] = std::make_unique<mlir_conversion::ConvConversionKernel>();
		registry_[OperatorType::GEMM] = std::make_unique<mlir_conversion::GemmConversionKernel>();
        registry_[OperatorType::MATMUL] = std::make_unique<mlir_conversion::MatmulConversionKernel>();
		registry_[OperatorType::MUL] = std::make_unique<mlir_conversion::MulConversionKernel>();
		registry_[OperatorType::RELU] = std::make_unique<mlir_conversion::ReluConversionKernel>();
	}

	auto MLIRConversionPass::convert() -> mlir::OwningOpRef<mlir::ModuleOp> {

        mlir_conversion::MLIRConversionData conversion_storage(context, builder, registry_, value_id_to_mlir_value, graph);

        context.getOrLoadDialect<mlir::func::FuncDialect>();
        context.getOrLoadDialect<mlir::arith::ArithDialect>();
        context.getOrLoadDialect<mlir::tensor::TensorDialect>();
        context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

        auto loc = builder.getUnknownLoc();

        mlir::ModuleOp module = mlir::ModuleOp::create(loc);

        builder.setInsertionPointToStart(module.getBody());

        mlir::FunctionType func_type = mlir_conversion::get_function_type(builder, graph);
        mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(loc, "main", func_type);
        func_op->setAttr("llvm.emit_c_interface", builder.getUnitAttr());

        mlir::Block* entry_block = func_op.addEntryBlock();

        builder.setInsertionPointToStart(entry_block);

        if (entry_block->getNumArguments() != graph.inputs.size()) {
            throw std::runtime_error("Conversion to MLIR: Wrong number of inputs in entry_block");
        }

        for (size_t i = 0; i < entry_block->getNumArguments(); ++i) {
            value_id_to_mlir_value[graph.inputs[i]] = entry_block->getArgument(i);
        }

        for (ValueID output : graph.outputs) {
            conversion_storage.convert_graph_value(output);
        }

        std::vector<mlir::Value> return_values;
        for (ValueID output : graph.outputs) {
            return_values.push_back(value_id_to_mlir_value[output]);
        }

        builder.create<mlir::func::ReturnOp>(loc, return_values);

        return mlir::OwningOpRef<mlir::ModuleOp>(module);
    }

}