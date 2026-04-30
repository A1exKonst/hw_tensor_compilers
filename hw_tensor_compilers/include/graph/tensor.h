#pragma once 
#include <vector>
#include "graph/shape.h"

namespace graph_engine {
	template <typename DType>
	class Tensor {
		std::vector<DType> data_;
		graph_engine::Shape shape_;

		Tensor(std::vector<DType> data__, const Shape& shape__) {
			data_ = std::move(data__);
			shape_ = shape__;
		};

		friend auto create_tensor(std::vector<DType> data__, const Shape& shape__) -> Tensor<DType>;
	public:
		Tensor() = default;
		Tensor(const Tensor&) = default;
		Tensor(Tensor&&) = default;
		~Tensor() = default;

		auto shape() const -> const graph_engine::Shape& { return shape_; };

		auto shape(graph_engine::Shape shape__) -> void;

		auto data() const -> const std::vector<DType>& { return data_; };
	};

	template <typename DType>
	auto create_tensor(std::vector<DType> data, const Shape& shape) -> Tensor<DType>;
}