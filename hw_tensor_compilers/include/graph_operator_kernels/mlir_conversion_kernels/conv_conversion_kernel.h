#pragma once
#include <unordered_map>
#include <memory>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/mlir_conversion_kernel.h"

#include "mlir/IR/Value.h"



namespace passes::mlir_conversion {

	class ConvConversionKernel : public MLIRConversionKernel {
	public:

		auto convert_graph_value(MLIRConversionData&, graph_engine::ValueID) -> mlir::Value override;

	};

}