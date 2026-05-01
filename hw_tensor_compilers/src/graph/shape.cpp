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
        if (rank__ >= MAX_VALUE_RANK) {
            throw std::length_error("Shape::rank(new_rank) : rank should be less than MAX_VALUE_RANK");
        };

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

    size_t Shape::elements_size() const {
        size_t sz = 1;
        for (unsigned short i = 0; i < rank_; ++i) sz*=dims[i];
        return sz;
    };

    std::optional<Shape> calculate_broadcast_compatible_shape(const Shape& s1, const Shape& s2, const unsigned start_rank) {

        unsigned min_rank = (s1.rank() < s2.rank()) ? s1.rank() : s2.rank();
        unsigned max_rank = (s1.rank() > s2.rank()) ? s1.rank() : s2.rank();
        const Shape& max_rank_shape = (s1.rank() > s2.rank()) ? s1 : s2;

        Shape result = Shape(max_rank);

        // compare shape in lower dims (higher indexes):
        for (int i = start_rank + 1; i <= min_rank; ++i) {
            auto first_dim = s1[s1.rank() - i];
            auto second_dim = s2[s2.rank() - i];
            bool is_compatible = ((first_dim == second_dim) ||
                (first_dim == 1) ||
                (second_dim == 1));
            if (!is_compatible) return std::nullopt;
            result[max_rank - i] = (first_dim > second_dim) ? first_dim : second_dim;
        }

        // broadcast shape in upper dims (lower indexes):
        for (int i = min_rank + 1; i <= max_rank; ++i) {
            result[max_rank - i] = max_rank_shape[max_rank - i];
        }
        return result;
    };

    std::optional<Shape> calculate_matmul_compatible_shape(const Shape& s1, const Shape& s2) {

        size_t s1_last_index = s1.rank() - 1;
        size_t s2_last_index = s2.rank() - 1;

        if (s1.rank() < 2 || s2.rank() < 2) return std::nullopt;
        if (s1[s1_last_index - 0] != s2[s2_last_index - 1]) return std::nullopt;

        // Broadcast upper dims:
        std::optional<Shape> broadcast = calculate_broadcast_compatible_shape(s1, s2, 2);
        if (!broadcast.has_value()) return std::nullopt;

        // Matmul op transforms:
        Shape result = broadcast.value();
        size_t last_index = result.rank() - 1;
        result[last_index - 0] = s2[s2_last_index - 0];
        result[last_index - 1] = s1[s1_last_index - 1];
        return result;
    };

    Shape transposed(const Shape& s, unsigned short axis_1, unsigned short axis_2) {
        Shape result{ s };

        int64_t k = std::move(result[axis_1]);
        result[axis_1] = std::move(result[axis_2]);
        result[axis_2] = std::move(k);

        return result;
    };

    Shape transposed(const Shape& s) {
        Shape result{ s };
        return transposed(result, result.rank() - 1, result.rank() - 2);
    };
};