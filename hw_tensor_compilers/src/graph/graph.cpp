#pragma once
#include "graph/graph.h"

#include <cassert>
#include <exception>
#include <optional>
#include <algorithm>

#include "io/out_graph_console.h"

namespace graph_engine {

    int64_t Shape::operator[](size_t i) const {
        if (i >= rank_) throw std::out_of_range("index >= shape::rank");
        return dims[i];
    };

    int64_t& Shape::operator[](size_t i) {
        if (i >= rank_) throw std::out_of_range("index >= shape::rank");
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

    bool Shape::operator== (const Shape& other) const {

        if (rank_ != other.rank()) return false;

        for (int i = 0; i < rank_; ++i) {
            if (dims[i] != other[i]) return false; 
        }
        return true;
    };

    std::optional<Shape> calculate_broadcast_compatible_shape(const Shape& s1, const Shape& s2) {

        unsigned min_rank = (s1.rank() < s2.rank()) ? s1.rank() : s2.rank();
        unsigned max_rank = (s1.rank() > s2.rank()) ? s1.rank() : s2.rank();
        const Shape& max_rank_shape = (s1.rank() > s2.rank()) ? s1 : s2;

        Shape result = Shape(max_rank);
        
        for (int i = 1; i < min_rank + 1; ++i) {
            auto first_dim = s1[s1.rank() - i];
            auto second_dim = s2[s2.rank() - i];
            bool is_compatible = ((first_dim == second_dim) || 
                                  (first_dim == 1) || 
                                  (second_dim == 1));
            if (!is_compatible) return std::nullopt;
            result[max_rank - i] = std::max(first_dim, second_dim);
        }

        for (int i = 0; i < max_rank - min_rank; ++i) {
            result[i] = max_rank_shape[i];
        }

        return result;
    };

    DataType math_result_data_type(DataType dt1, DataType dt2) {
        if ((dt1 == DataType::DELETED_VALUE) ||
            (dt2 == DataType::DELETED_VALUE)) {
            throw std::runtime_error("Tried to get math_result.dtype, but it is DataType::DELETED_VALUE");
        };
        if (static_cast<uint8_t>(dt1) < static_cast<uint8_t>(dt2)) return dt2;
        return dt1;
    };

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


