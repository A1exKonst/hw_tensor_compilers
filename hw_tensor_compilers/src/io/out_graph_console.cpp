#include "io/out_graph_console.h"

using namespace graph_engine;

std::ostream& operator<< (std::ostream& out, const AttributeValue& attr_val) {
    // using AttributeValue = std::variant<int, float, std::string, std::vector<int64_t>>;
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
            std::is_same_v<T, std::vector<float>>) {
            std::cout << "[vector size " << arg.size() << "]";
        }
        else {
            std::cout << arg; // float, int64_t and string
        }

        }, attr_val);
    return out;
};

std::ostream& operator<< (std::ostream& out, const Attributes& attrs) {
    // using Attributes = std::map<std::string, AttributeValue>;
    for (auto& [name, value] : attrs) out << "<" << name << ";" << value << ">, ";
    return out;
};

std::ostream& operator<< (std::ostream& out, OperatorType op) {
    out << operator_type_to_str.at(op);
    return out;
};

std::ostream& operator<< (std::ostream& out, const Node& node) {
    out << "Node " << node.op_type << "["
        << node.attr << "] (";
    for (auto id : node.inputs) out << "V" << id << ", ";
    out << ") -> (";
    for (auto id : node.outputs) out << "V" << id << ", ";
    out << ")";
    return out;
};

std::ostream& operator<< (std::ostream& out, DataType dt) {
    out << data_type_to_str.at(dt);
    return out;
};

std::ostream& operator<< (std::ostream& out, const Shape& shape) {
    out << "<";
    for (int i = 0; i < shape.rank() && i < MAX_VALUE_RANK; ++i) out << shape[i] << ",";
    out << ">";
    return out;
}

std::ostream& operator<< (std::ostream& out, const Value& value) {
    out << "Value " << value.dtype << value.shape << "[Node " << value.producer_node_id << "] -> (";
    for (auto id : value.consumer_node_ids) out << "N" << id << ", ";
    out << ")";
    return out;
};

std::ostream& operator<< (std::ostream& out, const Graph& graph) {
    out << "Graph" << std::endl;

    out << "Inputs: ";
    for (auto& input : graph.inputs) out << input << ",";
    out << std::endl;

    out << "Outputs: ";
    for (auto& output : graph.outputs) out << output << ",";
    out << std::endl;

    out << "Nodes:" << std::endl;
    for (int i = 0; i < graph.nodes.size(); ++i) {
        out << i << ":" << graph.nodes.at(i) << std::endl;
    };

    out << "Values:" << std::endl;
    for (int i = 0; i < graph.values.size(); ++i) {
        out << i << ":" << graph.values.at(i) << std::endl;
    };
    return out;
};