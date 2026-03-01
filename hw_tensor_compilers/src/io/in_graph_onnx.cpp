#include "io/in_graph_onnx.h"
#include <cassert>
#include <fstream>
#include <unordered_map>
#include "io/out_graph_console.h"
#include <iostream>
#include <variant>

using namespace graph_engine;

Graph io::import_from_model(const std::string& filename) {
    onnx::ModelProto protobuf_model;

    std::ifstream model_file{ filename, std::ios::ate | std::ios::binary };

    if (!model_file.is_open()) {
        throw std::runtime_error("Exception: could not open file " + filename);
    };

    std::ifstream input{ filename, std::ios::binary };

    protobuf_model.ParseFromIstream(&input);

    std::cout << "File: " << filename << std::endl;
    std::cout << "IR version: " << protobuf_model.ir_version() << std::endl;
    std::cout << "Producer: " << protobuf_model.producer_name() << std::endl << std::endl;

    return io::import_from_model(protobuf_model);
};

Graph io::import_from_model(const onnx::ModelProto& model) {
    Graph graph;
    const auto& onnx_graph = model.graph();

    std::vector<Node> nodes;
    std::vector<Value> values;
    std::vector<ValueID> inputs, outputs;

    graph.reserve(model.graph().node_size(),
                  model.graph().initializer_size());
    graph.inputs.reserve(model.graph().input_size());
    graph.outputs.reserve(model.graph().output_size());

    /*
    Initialization of name_to_value_id.
    Graph is a flat tree, so name_to_value_id performs as a map : value_name -> Value*
    */
    std::unordered_map<std::string, ValueID> name_to_value_id;

    // ================ add input Values =============================
    for (const onnx::ValueInfoProto& input : onnx_graph.input()) {
        ValueID val_id = convert_value_info(input, graph);
        name_to_value_id[input.name()] = val_id;
        graph.inputs.push_back(val_id);
    };

    // ================ add output Values =============================
    for (const onnx::ValueInfoProto& output : onnx_graph.output()) {
        ValueID val_id = convert_value_info(output, graph);
        name_to_value_id[output.name()] = val_id;
        graph.outputs.push_back(val_id);
    };

    // ================ add weight Values ============================
    for (const auto& initializer : onnx_graph.initializer()) {
        NodeID new_node_id = graph.nodes.size();
        Node new_weight_node{ OperatorType::CONSTANT, {}, {}, Attributes() };

        Shape shape;
        shape.rank(initializer.dims().size());
        for (int i = 0; i < initializer.dims().size(); ++i) { shape[i] = initializer.dims().at(i); };

        int32_t onnx_dtype = initializer.data_type();
        DataType dtype = map_dtype(onnx_dtype);
        ValueID new_value_id = graph.add_value(std::move(shape), dtype, new_node_id);

        new_weight_node.outputs.push_back(new_value_id);
        graph.nodes.push_back(new_weight_node);

        name_to_value_id[initializer.name()] = new_value_id;
    };

    // ================ add Nodes ====================================
    for (const auto& onnx_node : onnx_graph.node()) {

        // parse Node inputs
        std::vector<ValueID> inputs;
        for (const auto& input_value_name : onnx_node.input()) {
            if (name_to_value_id.contains(input_value_name)) {
                ValueID input_value_id = name_to_value_id.at(input_value_name);
                inputs.push_back(input_value_id);
            }
            else { throw std::runtime_error("Input Value expected, but no such value_name found: " + input_value_name); };
        };

        // parse Node attributes
        Attributes attrs;
        for (const auto& attr : onnx_node.attribute()) {
            attrs[attr.name()] = parse_attribute(attr);
        };

        // add parsed Node to Graph
        const std::string& op_type_name = onnx_node.op_type();
        OperatorType op_type = map_operator_type(op_type_name);
        NodeID new_node_id = graph.add_node(op_type, inputs, {}, attrs);

        // parse Node outputs
        for (const auto& output_value_name : onnx_node.output()) {

            ValueID output_value_id;
            if (name_to_value_id.contains(output_value_name)) output_value_id = name_to_value_id.at(output_value_name);
            else output_value_id = graph.add_value({}, DataType::UNDEFINED, new_node_id);

            graph.nodes[new_node_id].outputs.push_back(output_value_id);
            name_to_value_id[output_value_name] = output_value_id;
        }
    }

    return graph;
};

DataType io::map_dtype(int32_t onnx_type) {
    switch (onnx_type) {
    case onnx::TensorProto_DataType_FLOAT:  return DataType::FLOAT32;
    case onnx::TensorProto_DataType_INT64:  return DataType::INT64;
    case onnx::TensorProto_DataType_BOOL:   return DataType::BOOL;
    default:                                return DataType::UNDEFINED;
    };
};

OperatorType io::map_operator_type(const std::string& op) {

    if (!str_to_operator_type.contains(op)) {
        throw std::runtime_error("No corresponding OperatorType found: '" + op + "'");
    };

    return str_to_operator_type.at(op);
};

/*
AttributeValue io::parse_attribute(const onnx::AttributeProto& attr) {
    //std::cout << "attributes: " <<attr.i() << " " << attr.f() << " " << attr.s() << std::endl;
    if (attr.has_f()) return attr.f();
    if (attr.has_i()) return (int)attr.i();
    if (attr.has_s()) return attr.s();
    if (attr.ints_size() > 0) {
        std::vector<int64_t> values;
        for (auto i : attr.ints()) values.push_back(i);
        return values;
    }
    return 0;
};
*/

AttributeValue io::parse_attribute(const onnx::AttributeProto& attr) {
    /*
    if (attr.name() == "alpha") {
        std::cout << "DEBUG: alpha type is " << attr.type()
            << " i=" << attr.i() << " f=" << attr.f() << std::endl;
    }
    */
    switch (attr.type()) {
    case onnx::AttributeProto::FLOAT:
        return attr.f(); // Возвращает float
    case onnx::AttributeProto::INT:
        return attr.i(); // Возвращает int64
    case onnx::AttributeProto::STRING:
        return attr.s(); // Возвращает string
    case onnx::AttributeProto::FLOATS: {
        std::vector<float> values;
        for (auto f : attr.floats()) values.push_back(f);
        return values;
    }
    case onnx::AttributeProto::INTS: {
        std::vector<int64_t> values;
        for (auto i : attr.ints()) values.push_back(i);
        return values;
    }
                                   // Добавь TENSOR для весов, если нужно
    default:
        return 0;
    }
}


ValueID io::convert_value_info(const onnx::ValueInfoProto& info, Graph& g) {
    Shape shape;
    auto onnx_shape = info.type().tensor_type().shape();
    
    shape.rank(onnx_shape.dim_size());
    for (int i = 0; i < onnx_shape.dim_size(); ++i) {
        shape[i] = onnx_shape.dim(i).dim_value();
    }
    return g.add_value(std::move(shape), map_dtype(info.type().tensor_type().elem_type()), size_t(-1));
};