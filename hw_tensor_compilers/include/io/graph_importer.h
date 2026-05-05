#pragma once
#include "graph/graph.h"



namespace io {
    /**
    * Interface class of different possible imports for class graph_engine::Graph
    */
    class GraphImporter {
    public:
        GraphImporter() = default;
        virtual ~GraphImporter() = default;
        GraphImporter(const GraphImporter&) = delete;
        GraphImporter& operator=(const GraphImporter&) = delete;

        virtual auto import_graph() -> graph_engine::Graph = 0;

    };
};