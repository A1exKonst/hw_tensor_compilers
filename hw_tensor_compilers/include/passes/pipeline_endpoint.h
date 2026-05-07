#pragma once

namespace passes {
	enum class PipelineEndpoint {
		GRAPH_INPUT,
		GRAPH_PASSES,
		MLIR_GENERATION,
		MLIR_LOWERING,
		EXECUTION
	};
}