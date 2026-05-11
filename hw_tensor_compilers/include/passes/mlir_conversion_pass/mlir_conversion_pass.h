#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"



namespace passes {

	class MLIRConversionPass {
	public:
		MLIRConversionPass(const graph_engine::Graph& graph, mlir::MLIRContext& context); // fill registry_
		~MLIRConversionPass() = default;

		MLIRConversionPass(const MLIRConversionPass& other) = default;
		MLIRConversionPass(MLIRConversionPass&& other) = default;

		MLIRConversionPass& operator=(const MLIRConversionPass&) = default;
		MLIRConversionPass& operator=(MLIRConversionPass&&) = default;

		[[nodiscard]]
		auto convert() -> mlir::OwningOpRef<mlir::ModuleOp>;

	private:
		std::unordered_map<
			graph_engine::OperatorType, 
			std::unique_ptr<mlir_conversion::MLIRConversionKernel>
		> registry_;

		const graph_engine::Graph& graph;

		mlir::MLIRContext& context;

		mlir::OpBuilder builder;

		std::unordered_map<graph_engine::ValueID, mlir::Value> value_id_to_mlir_value;

	};

}