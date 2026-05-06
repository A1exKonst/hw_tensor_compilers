#pragma once
#include "graph/graph.h"

#include <cassert>
#include <exception>
#include <optional>
#include <algorithm>

#include "io/console_graph_exporter.h"

namespace graph_engine {

    void Graph::reserve(size_t nodes_count, size_t values_count) {
        nodes.reserve(nodes_count);
        values.reserve(values_count);
    };

    bool Graph::is_graph_valid() const { return true; };


    NodeID Graph::add_node(
        OperatorType type, 
        const std::vector<ValueID>& inputs, 
        const std::vector<ValueID>& outputs,
        Attributes attrs
    ) {
        NodeID node_id = nodes.size();
        nodes.push_back({ type, inputs, {}, std::move(attrs)});

        // Обновляем связи у входных значений
        for (ValueID input_val_id : inputs) {
            values[input_val_id].consumer_node_ids.push_back(node_id);
        };

        return node_id;
    };

    ValueID Graph::add_value(Shape shape, DataType dtype, NodeID producer_id) {
        ValueID val_id = ValueID(values.size());
        values.emplace_back(Value(std::move(shape),
                                            dtype,
                                            producer_id,
                                            {}      // consumer_node_ids
        ));
        return val_id;
    };
};


