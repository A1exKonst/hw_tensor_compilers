#pragma once
#include <unordered_map>
#include "onnx.pb.h"
#include "graph/graph.h"

namespace io {
    inline const std::unordered_map<std::string, graph_engine::OperatorType> str_to_operator_type = {
        {"Conv", graph_engine::OperatorType::CONV}, {"Relu", graph_engine::OperatorType::RELU},
        {"MatMul", graph_engine::OperatorType::MATMUL}, {"Add", graph_engine::OperatorType::ADD},
        {"Mul", graph_engine::OperatorType::MUL},{"Gemm", graph_engine::OperatorType::GEMM},
        {"Constant", graph_engine::OperatorType::CONSTANT}
    };

    graph_engine::Graph import_from_model(const onnx::ModelProto& model);

    graph_engine::Graph import_from_model(const std::string& filename);

    // mapping onnx_type -> DataType
    graph_engine::DataType map_dtype(int32_t onnx_type);

    // mapping string -> OperatorType
    graph_engine::OperatorType map_operator_type(const std::string& op);

    // convert onnx::AttributeProto -> attributes.h::AttributeValue
    graph_engine::AttributeValue parse_attribute(const onnx::AttributeProto& attr);

    // add Value from onnx::ModelProto to Graph
    graph_engine::ValueID convert_value_info(const onnx::ValueInfoProto& info, graph_engine::Graph& g);
};
