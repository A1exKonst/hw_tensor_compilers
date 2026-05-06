#pragma once
#include <string>

#include "graph/graph.h"



namespace io {
    /**
    * Interface class of different possible exports for class graph_engine::Graph
    */
    class GraphExporter {
    public:
        virtual ~GraphExporter() = default;
        GraphExporter(const GraphExporter&) = delete;
        GraphExporter& operator=(const GraphExporter&) = delete;

        virtual auto operator<<(const graph_engine::Graph& graph) -> GraphExporter& = 0;

    protected:
        GraphExporter() = default;

        /* 
        template <typename T>
        auto operator<<(const T& data) -> GraphExporter& {
            write(std::to_string(data));
            return *this;
        }

        GraphExporter& operator<<(const std::string& data) {
            write(data);
            return *this;
        }

        GraphExporter& operator<<(const char* data) {
            write(data);
            return *this;
        }
        protected:
        virtual void write(const std::string& s) = 0;
        */
    };

};