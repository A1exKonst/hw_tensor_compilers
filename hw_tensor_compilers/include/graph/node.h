#pragma once
#include <vector>
#include <optional>
#include <unordered_map>

#include "attributes.hpp"


namespace graph_engine {
    using ValueID = size_t;

    enum class OperatorType {
        ADD,
        MUL,
        CONV,
        RELU,
        MATMUL,
        GEMM,
        
        DTYPE_CONVERSION,
        CONSTANT,
        INPUT
    };

    // Узел графа (Операция)
    class Node {
    public:
        OperatorType op_type;

        std::vector<ValueID> inputs;  // ID входящих Value
        std::vector<ValueID> outputs; // ID исходящих Value

        Attributes attr;

        Node() = default;
        Node(const Node&) = default;
        Node(Node&&) = default;
        ~Node() = default;
        Node& operator=(const Node&) = default;
        Node& operator=(Node&&) = default;

        Node(OperatorType op_type_, std::vector<ValueID> inputs_, std::vector<ValueID> outputs_, Attributes attr_) :
            op_type(op_type_), inputs(std::move(inputs_)), outputs(std::move(outputs_)),  attr(attr_) {
        };
    };

    inline const std::unordered_map<OperatorType, std::string> operator_type_to_str = {
        {graph_engine::OperatorType::ADD,   "Add"},     {graph_engine::OperatorType::CONSTANT,  "Const"},
        {graph_engine::OperatorType::CONV, "Conv"},    {graph_engine::OperatorType::INPUT,     "Input"},
        {graph_engine::OperatorType::MATMUL,"Matmul"},  {graph_engine::OperatorType::RELU,      "ReLu"},
        {graph_engine::OperatorType::MUL,"Mul"},        {graph_engine::OperatorType::GEMM,      "Gemm"},

    };
}