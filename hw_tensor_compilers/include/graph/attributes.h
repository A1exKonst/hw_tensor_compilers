#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <stdexcept>



namespace graph_engine {

    // int64_t is stated in ONNX standard, thus it is used here
    using AttributeValue = std::variant<
        int64_t, 
        float, 
        std::string, 
        std::vector<int64_t>, 
        std::vector<float>
    >;

    using Attributes = std::unordered_map<std::string, AttributeValue>;

}