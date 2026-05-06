#pragma once 
#include <vector>

#include "graph/shape.h"



namespace graph_engine {
	template <typename DType>
	class Tensor;

	template <typename DType>
	auto make_tensor(std::vector<DType> data, Shape shape) -> Tensor<DType>;

	template <typename DType>
	class Tensor {
		std::vector<DType> data_;
		graph_engine::Shape shape_;

		Tensor(std::vector<DType> data__, Shape shape__) {
			data_ = std::move(data__);
			shape_ = std::move(shape__);
		};

		friend auto make_tensor(std::vector<DType> data__, Shape shape__) -> Tensor<DType>;
	public:
		Tensor() = default;
		Tensor(const Tensor&) = default;
		Tensor(Tensor&&) = default;
		~Tensor() = default;

		auto shape() const -> const graph_engine::Shape& { return shape_; };

		auto shape(graph_engine::Shape shape__) -> void {
			if (data_.size() != shape__.elements_size()) {
				throw std::length_error("Invalid Tensor.shape(new_shape) : data.size() != new_shape.elements_size()");
			}
			shape_ = shape__;
		};

		auto data() -> std::vector<DType>& { return data_; };

		static auto make_tensor(std::vector<DType> data, Shape shape) -> Tensor<DType> {
			if (data.size() != shape.elements_size()) {
				throw std::length_error("Invalid Tensor initialization : data.size() != shape.elements_size()");
			}
			return Tensor<DType>(std::move(data), std::move(shape));
		};
	};
}