#pragma once
#include <unordered_map>
#include <string>

#include "graph/node.h"



namespace io {

    inline const std::unordered_map<std::string, graph_engine::OperatorType> str_to_operator_type = {
        {"Conv", graph_engine::OperatorType::CONV}, {"Relu", graph_engine::OperatorType::RELU},
        {"MatMul", graph_engine::OperatorType::MATMUL}, {"Add", graph_engine::OperatorType::ADD},
        {"Mul", graph_engine::OperatorType::MUL},{"Gemm", graph_engine::OperatorType::GEMM},
        {"Constant", graph_engine::OperatorType::CONSTANT}
    };

}

