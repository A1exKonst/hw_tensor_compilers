#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/mlir_conversion_kernel.h"

using namespace graph_engine;



namespace passes::mlir_conversion {

	auto MLIRConversionData::convert_graph_value(graph_engine::ValueID value_id) -> mlir::Value {

		// check if already done when recurringly call conversion
		if (value_id_to_mlir_value.find(value_id) != value_id_to_mlir_value.end()) {
			return value_id_to_mlir_value.at(value_id);
		}

		OperatorType producer_type = graph.nodes[graph.values[value_id].producer_node_id].op_type;
		mlir::Value result = registry_[producer_type]->convert_graph_value(*this, value_id);
		value_id_to_mlir_value[value_id] = result;
		return result;
	}

}