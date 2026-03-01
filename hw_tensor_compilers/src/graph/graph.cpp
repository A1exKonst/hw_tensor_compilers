#pragma once
#include "graph/graph.h"
#include <cassert>
#include "io/out_graph_console.h"

namespace graph_engine {

    int64_t Shape::operator[](size_t i) const {
        assert(i < rank_);
        return dims[i];
    };

    int64_t& Shape::operator[](size_t i) {
        assert(i < rank_);
        return dims[i];
    };

    void Shape::rank(size_t rank__) {
        assert(rank__ < MAX_VALUE_RANK);

        // when expanding matrix default size in a dim is 1
        for (unsigned short i = rank_; i < rank__; ++i) dims[i] = 1;

        rank_ = rank__;

        // ensure cropping shape when flattenning matrix
        for (unsigned int i = rank_ + 1; i < MAX_VALUE_RANK; ++i) dims[i] = 0;
    };

    size_t Shape::rank() const noexcept { return rank_;};

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


