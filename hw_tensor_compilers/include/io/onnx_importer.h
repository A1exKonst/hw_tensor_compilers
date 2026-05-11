#pragma once
#include <string>

#include "io/graph_importer.h"
#include "graph/graph.h"



namespace io {

    /**
    * Importer for class graph_engine::Graph
    * from files in format "filename.onnx"
    */
    class OnnxImporter : public GraphImporter {
    private:
        std::string filename_;

    public:
        explicit OnnxImporter(std::string new_filename) 
            : filename_(std::move(new_filename)) {}
        ~OnnxImporter() override = default;

        OnnxImporter(const OnnxImporter&) = delete;
        OnnxImporter& operator=(const OnnxImporter&) = delete;

        OnnxImporter(OnnxImporter&&) = default;
        OnnxImporter& operator=(OnnxImporter&&) = default;

        [[nodiscard]]
        auto import_graph() -> graph_engine::Graph override;
    };

}