#pragma once

namespace passes {
	enum class PipelineEndpoint {
		GRAPH_INPUT,
		SEMANTICS_INFERER,
		MLIR_GENERATION,
		MLIR_LOWERING,
		EXECUTION
	};
}