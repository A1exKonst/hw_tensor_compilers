#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>



namespace graph_engine {

    // Shape is a shape (or dims) of a given Value : Value.shape
    class Shape;

    inline constexpr size_t MAX_VALUE_RANK = 10;

    class Shape {
    protected:
        std::array<int64_t, MAX_VALUE_RANK> dims{ 0,0,0,0,0,0,0,0,0,0 };
        // std::array for cache locality, as shape is usually less than 8

        size_t rank_ = 0;
        // current rank of Value

        Shape(std::initializer_list<int64_t> list) {
            std::copy(list.begin(),
                list.size() > 10 ? list.begin() + 10 : list.end(),
                dims.begin()
            );
            rank_ = list.size();
        };

        //friend Shape make_shape(std::initializer_list<int64_t> list);

    public:
        Shape() : rank_(0), dims({ 0,0,0,0,0,0,0,0,0,0 }) {};
        Shape(const Shape&) = default;
        Shape(Shape&&) noexcept = default;
        ~Shape() = default;
        Shape& operator=(const Shape&) = default;
        Shape& operator=(Shape&&) = default;

        bool operator== (const Shape&) const; // = default; (min c++20)

        Shape(size_t rank__) {
            rank(rank__);
        };

        int64_t operator[](size_t i) const;

        int64_t& operator[](size_t i);

        void rank(size_t rank__);

        size_t rank() const noexcept;

        size_t elements_size() const;

        static auto make_shape(std::initializer_list<int64_t> list) -> Shape {
            if (list.size() > MAX_VALUE_RANK) {
                throw std::length_error("Shape::Shape(std::initializer_list<int64_t>) : list.size() > MAX_VALUE_RANK");
            };
            return Shape(list);
        };


        // std container interface:
        auto begin() const noexcept { return dims.data(); }
        auto end() const noexcept { return dims.data() + rank_; }
        size_t size() const noexcept { return rank_; }
        using value_type = int64_t;
        using const_iterator = const int64_t*;
        using iterator = const_iterator;
    };

    auto calculate_broadcast_compatible_shape(const Shape& s1, const Shape& s2, const unsigned start_rank = 0) -> std::optional<Shape>;

    auto calculate_matmul_compatible_shape(const Shape& s1, const Shape& s2) -> std::optional<Shape>;

    auto transposed(const Shape& s, unsigned short axis_1, unsigned short axis_2) -> Shape;

    auto transposed(const Shape& s) -> Shape;
};
