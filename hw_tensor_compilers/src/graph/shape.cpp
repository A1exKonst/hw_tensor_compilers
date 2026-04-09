#pragma once
#include "graph/shape.h"

#include <cassert>
#include <exception>
#include <stdexcept>
#include <optional>
#include <algorithm>

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

    size_t Shape::rank() const noexcept { return rank_; };

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
            result[max_rank - i] = (first_dim > second_dim) ? first_dim : second_dim;
        }

        for (int i = 0; i < max_rank - min_rank; ++i) {
            result[i] = max_rank_shape[i];
        }

        return result;
    };
};