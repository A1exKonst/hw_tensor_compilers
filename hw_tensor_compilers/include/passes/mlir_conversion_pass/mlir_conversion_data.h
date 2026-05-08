#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"



namespace passes::mlir_conversion {

	class MLIRConversionData;

	class MLIRConversionKernel;

	using ConversionRegistryType = std::unordered_map<
		graph_engine::OperatorType,
		std::unique_ptr<MLIRConversionKernel>>;

	/*
	* Data holder for conversion of graph_engine::Graph to mlir::ModuleOp
	*/
	class MLIRConversionData {
	public:
		MLIRConversionData(
			mlir::MLIRContext& new_context, mlir::OpBuilder& new_builder, 
			ConversionRegistryType& new_registry, const graph_engine::Graph& new_graph) : 
			context(new_context), builder(new_builder), 
			registry_(new_registry), graph(new_graph) {
		};
		~MLIRConversionData() = default;

		MLIRConversionData(const MLIRConversionData& other) = default;
		MLIRConversionData& operator=(const MLIRConversionData&) = default;

		MLIRConversionData(MLIRConversionData&& other) = default;
		MLIRConversionData& operator=(MLIRConversionData&&) = default;

		mlir::MLIRContext& context;

		mlir::OpBuilder& builder;

		const graph_engine::Graph& graph;

		std::unordered_map<graph_engine::ValueID, mlir::Value> value_id_to_mlir_value;

		auto convert_graph_value(graph_engine::ValueID) -> mlir::Value;

	protected:
		ConversionRegistryType& registry_;

	};

}