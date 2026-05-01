#pragma once
#include <string>

#include "passes/pipeline_endpoint.h"



namespace passes {
	class PassesPipeline {
	public:
		static void apply_pipeline(const std::string& filename, passes::PipelineEndpoint endpoint, bool debug = false);
	};
}