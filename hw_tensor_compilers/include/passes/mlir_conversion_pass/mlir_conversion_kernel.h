#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"



namespace passes {
	namespace mlir_conversion {

		class MLIRConversionPass;

		class MLIRConversionData;

		/*
		* Interface class for conversion graph_engine::Graph to mlir::ModuleOp
		*/
		class MLIRConversionKernel {
		public:
			virtual ~MLIRConversionKernel() = default;

			MLIRConversionKernel(const MLIRConversionKernel& other) = delete;
			MLIRConversionKernel& operator=(const MLIRConversionKernel&) = delete;

			virtual auto convert_graph_value(MLIRConversionData&, graph_engine::ValueID) -> mlir::Value = 0;

		protected:
			MLIRConversionKernel() = default;

		};

	}

}