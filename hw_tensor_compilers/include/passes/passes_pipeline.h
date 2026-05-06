#pragma once
#include <vector>

#include "passes/pipeline_endpoint.h"
#include "passes/graph_pass.h"
#include "io/graph_importer.h"
#include "io/graph_exporter.h"



namespace passes {

    class PassesPipeline {
    public:
        PassesPipeline(io::GraphImporter& new_importer,
            io::GraphExporter& new_exporter,
            std::vector<std::unique_ptr<GraphPass>> new_graph_passes)
            : importer(new_importer),
            exporter(new_exporter),
            graph_passes(std::move(new_graph_passes)) {
        }

        PassesPipeline(const PassesPipeline&) = delete;
        PassesPipeline& operator=(const PassesPipeline&) = delete;

        PassesPipeline(PassesPipeline&&) = default;
        PassesPipeline& operator=(PassesPipeline&&) = default;

        void apply_pipeline(passes::PipelineEndpoint endpoint, bool debug = false);

    private:
        io::GraphImporter& importer;
        io::GraphExporter& exporter;
        std::vector<std::unique_ptr<GraphPass>> graph_passes;
    };

}