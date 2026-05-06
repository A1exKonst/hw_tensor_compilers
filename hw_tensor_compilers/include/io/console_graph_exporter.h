#pragma once
#include "graph/graph.h"
#include "io/graph_exporter.h"



namespace io {
    
    /*
    * Exporter for class graph_engine::Graph
    * dumps all Graph info into console
    */
    class ConsoleGraphExporter : public GraphExporter {
    public:
        ConsoleGraphExporter() {
        };
        ~ConsoleGraphExporter() override = default;

        ConsoleGraphExporter(const ConsoleGraphExporter&) = delete;
        ConsoleGraphExporter& operator=(const ConsoleGraphExporter&) = delete;

        ConsoleGraphExporter(ConsoleGraphExporter&&) = default;
        ConsoleGraphExporter& operator=(ConsoleGraphExporter&&) = default;

        auto operator<<(const graph_engine::Graph& graph) -> GraphExporter& override;

    };

};

auto operator<< (std::ostream& out, const graph_engine::AttributeValue& attr_val) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::Attributes& attrs) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::OperatorType& op) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::Node& node) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::DataType& dt) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::Shape& shape) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::Value& value) -> std::ostream&;

auto operator<< (std::ostream& out, const graph_engine::Graph& graph) -> std::ostream&;