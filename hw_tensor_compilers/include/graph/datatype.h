#pragma once
#include <cstdint>
#include <unordered_map>
#include <string>

namespace graph_engine {

    enum class DataType : uint8_t;

    enum class DataType : uint8_t {
        DELETED_VALUE = 0,

        UNDEFINED = 1,
        FLOAT32 = 2,
        INT64 = 3,
        BOOL = 4,
    };

    DataType math_result_data_type(DataType dt1, DataType dt2);

    inline const std::unordered_map<DataType, std::string> data_type_to_str = {
        {graph_engine::DataType::BOOL,"BOOL"},  {graph_engine::DataType::FLOAT32,"FLOAT32"},
        {graph_engine::DataType::INT64,"INT64"},{graph_engine::DataType::UNDEFINED,"UNDEF_DTYPE"}
    };
};
