#include <stdexcept>

#include "graph/datatype.h"



namespace graph_engine {

    auto math_result_data_type(DataType dt1, DataType dt2) -> DataType {
        if ((dt1 == DataType::DELETED_VALUE) ||
            (dt2 == DataType::DELETED_VALUE)) {
            throw std::runtime_error("Tried to get math_result.dtype, but it is DataType::DELETED_VALUE");
        }
        if (static_cast<uint8_t>(dt1) < static_cast<uint8_t>(dt2)) return dt2;
        return dt1;
    }

}