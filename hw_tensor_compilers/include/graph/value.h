#pragma once
#include <vector>
#include <cstdint>

#include "graph/shape.h"
#include "graph/datatype.h"



namespace graph_engine {
    using NodeID = size_t;

    // Each operation in a computing graph is a Node (declared in "graph/node.h").
    // Result of each operation is Value (or alias Tensor)
    // Value is a descriptor of raw data (if given).
    class Value;

    // Shape is a shape (or dims) of a given Value : Value.shape
    class Shape;
    
    enum class DataType : uint8_t;

    class Value {
    public:
        Shape shape;
        DataType dtype = DataType::UNDEFINED;

        NodeID producer_node_id;            // Node ID, which operation has this Value as a result of itself
        // producer_node can be only one, this is a part of SSA for tensor compilers.

        std::vector<NodeID> consumer_node_ids; // vector of Node ID's, which use this Value (or alias Tensor)
        // optionally SmallVector, but it is a studying project, so std::vector was left.

        Value(const Value&) = default;
        Value(Value&&) noexcept = default;
        ~Value() = default;
        Value& operator=(const Value&) = default;
        Value& operator=(Value&&) = default;

        Value(Shape shape_, DataType dtype_, NodeID producer_node_id_, std::vector<NodeID> consumer_node_ids_) :
            shape(std::move(shape_)), dtype(std::move(dtype_)), producer_node_id(producer_node_id_),
            consumer_node_ids(std::move(consumer_node_ids_)){};
    };
};
