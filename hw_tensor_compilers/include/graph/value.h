#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <iostream>

namespace graph_engine {
    using NodeID = size_t;

    // Each operation in a computing graph is a Node (declared in "graph/node.h").
    // Result of each operation is Value (or alias Tensor)
    // Value is a descriptor of raw data (if given).
    class Value;

    // Shape is a shape (or dims) of a given Value : Value.shape
    class Shape;

    enum class DataType : uint8_t;

    enum class DataType : uint8_t {
        DELETED_VALUE = 0,

        UNDEFINED = 1,
        FLOAT32 = 2,
        INT64 = 3,
        BOOL = 4,
    };

    inline const std::unordered_map<DataType, std::string> data_type_to_str = {
        {graph_engine::DataType::BOOL,"BOOL"},  {graph_engine::DataType::FLOAT32,"FLOAT32"},
        {graph_engine::DataType::INT64,"INT64"},{graph_engine::DataType::UNDEFINED,"UNDEF_DTYPE"}
    };

    inline constexpr size_t MAX_VALUE_RANK = 10;

    class Shape {
    protected:
        std::array<int64_t, MAX_VALUE_RANK> dims{0,0,0,0,0,0,0,0,0,0};
        // std::array for cache locality, as shape is usually less than 8

        size_t rank_ = 0;
        // current rank of Value

    public:
        Shape() : rank_(0), dims({ 0,0,0,0,0,0,0,0,0,0 }) {};
        Shape(const Shape&) = default;
        Shape(Shape&&) = default;
        ~Shape() = default;
        Shape& operator=(const Shape&) = default;
        Shape& operator=(Shape&&) = default;

        bool operator== (const Shape&) const = default;

        Shape(size_t rank__) {
            std::cout << std::endl << "received rank: " << rank__;
            rank(rank__); 
        };

        int64_t operator[](size_t i) const;

        int64_t& operator[](size_t i);

        void rank(size_t rank__);

        size_t rank() const noexcept;
    };

    class Value {
    public:
        Shape shape;
        DataType dtype = DataType::UNDEFINED;

        NodeID producer_node_id;            // Node ID, which operation has this Value as a result of itself
        // producer_node can be only one, this is a part of SSA for tensor compilers.

        std::vector<NodeID> consumer_node_ids; // vector of Node ID's, which use this Value (or alias Tensor)
        // optionally SmallVector, but it is a studying project, so std::vector was left.

        Value(const Value&) = default;
        Value(Value&&) = default;
        ~Value() = default;
        Value& operator=(const Value&) = default;
        Value& operator=(Value&&) = default;

        Value(Shape shape_, DataType dtype_, NodeID producer_node_id_, std::vector<NodeID> consumer_node_ids_) :
            shape(std::move(shape_)), dtype(std::move(dtype_)), producer_node_id(producer_node_id_),
            consumer_node_ids(std::move(consumer_node_ids_)){};
    };
};
