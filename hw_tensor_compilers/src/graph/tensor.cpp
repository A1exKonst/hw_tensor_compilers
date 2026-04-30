#include "graph/tensor.h"
#include <stdexcept>

namespace graph_engine {

	template <typename DType>
	auto create_tensor(std::vector<DType> data, const Shape& shape) -> Tensor<DType> {
		if (data.size() != shape.elements_size()) {
			throw std::length_error("Invalid Tensor initialization : data.size() != shape.elements_size()");
		}
		return Tensor<DType>(std::move(data), std::move(shape));
	};
}