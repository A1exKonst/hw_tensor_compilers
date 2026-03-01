#pragma once
#include <vector>

#include "value.h"
#include "node.h"
#include "attributes.hpp"

namespace graph_engine {
    class Graph {
    public:
        std::vector<Value> values;
        std::vector<Node> nodes;

        std::vector<ValueID> inputs;
        std::vector<ValueID> outputs;

        Graph() = default;
        Graph(const Graph&) = default;
        Graph(Graph&&) = default;
        ~Graph() = default;

        Graph(std::vector<Node> nodes_, std::vector<Value> values_, 
            std::vector<ValueID> inputs_, std::vector<ValueID> outputs_) : 
            nodes(std::move(nodes_)), values(std::move(values_)),
            inputs(std::move(inputs_)), outputs(std::move(outputs_)) {

            if (!is_graph_valid()) {
                throw std::runtime_error("Graph : graph validity check was not passed");
            };
        };

        bool is_graph_valid() const;

        void reserve(size_t nodes_count, size_t values_count);

        // add Node without checking if Graph will still be valid
        // NodeID add_node(OperatorType type, const std::vector<ValueID>& inputs, Attributes attrs);

        NodeID add_node(OperatorType type, const std::vector<ValueID>& inputs, 
            const std::vector<ValueID>& outputs, Attributes attrs);

        // add Value without checking if Graph will still be valid
        ValueID add_value(Shape shape, DataType dtype, NodeID producer_id);
    };
};
