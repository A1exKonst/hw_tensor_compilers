#include "passes/mlir_converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace graph_engine;

using namespace passes;


auto GraphToMLIRConverter::datatype_to_mlir_type(mlir::OpBuilder& builder, graph_engine::DataType dtype) -> mlir::Type {
    mlir::Type return_type;
    switch (dtype) {
    case DataType::BOOL:
        return_type = builder.getI1Type();
        break;
    case DataType::FLOAT32:
        return_type = builder.getF32Type();
        break;
    case DataType::INT64:
        return_type = builder.getI64Type();
        break;
    default:
        throw std::runtime_error("Invalid Datatype encountered when converting to mlir");
    }
    return return_type;
}

auto GraphToMLIRConverter::get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, ValueID value_id) -> mlir::RankedTensorType {
    mlir::Type dtype = datatype_to_mlir_type(builder, graph.values[value_id].dtype);
    const Shape& s = graph.values[value_id].shape;
    llvm::ArrayRef<int64_t> shape_slice(&*s.begin(), s.rank());
    return mlir::RankedTensorType::get(shape_slice, dtype);
};

auto GraphToMLIRConverter::get_function_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph)->mlir::FunctionType {
    std::vector<mlir::Type> inputs;
    std::vector<mlir::Type> outputs;

    inputs.reserve(graph.inputs.size());
    outputs.reserve(graph.outputs.size());

    std::transform(graph.inputs.begin(), graph.inputs.end(), std::back_inserter(inputs),
        [&](ValueID v) { return get_value_tensor_type(builder, graph, v); });
    std::transform(graph.outputs.begin(), graph.outputs.end(), std::back_inserter(outputs),
        [&](ValueID v) { return get_value_tensor_type(builder, graph, v); });

    return builder.getFunctionType(inputs, outputs);
};

template <typename IntOp, typename FloatOp>
mlir::Value create_mlir_binary_operation(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::Value lhs, mlir::Value rhs) {
    if (lhs.getType().isa<mlir::FloatType>()) {
        return builder.create<FloatOp>(loc, lhs, rhs).getResult();
    }
    return builder.create<IntOp>(loc, lhs, rhs).getResult();
}

template <typename IntOp, typename FloatOp>
mlir::Value GraphToMLIRConverter::create_binary_operation(NodeID producer) {
    mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer, 0);
    mlir::Value lhs = value_id_to_mlir_value[graph.nodes[producer].inputs[0]];
    mlir::Value rhs = value_id_to_mlir_value[graph.nodes[producer].inputs[1]];

    return create_mlir_binary_operation<IntOp, FloatOp>(builder, loc, lhs, rhs);
};

auto GraphToMLIRConverter::convert_value_to_mlir_value(graph_engine::ValueID value) -> mlir::Value {
    if (value_id_to_mlir_value.find(value) != value_id_to_mlir_value.end()) {
        return value_id_to_mlir_value.at(value);
    }

    mlir::Value result;

    NodeID producer_node = graph.values[value].producer_node_id;

    mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, 0);

    for (ValueID input : graph.nodes[producer_node].inputs) { convert_value_to_mlir_value(input); }

    switch (graph.nodes[producer_node].op_type) {
    case OperatorType::ADD:
        result = create_binary_operation<mlir::arith::AddIOp, mlir::arith::AddFOp>(producer_node);
        break;
    case OperatorType::MUL:
        result = create_binary_operation<mlir::arith::MulIOp, mlir::arith::MulFOp>(producer_node);
        break;
    case OperatorType::CONSTANT:{
        auto& weights = std::get<std::vector<float>>(graph.nodes[producer_node].attr.at("weights"));
        auto tensor_type = get_value_tensor_type(builder, graph, value);
        auto weights_attr = mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(weights));

        auto constant_op = builder.create<mlir::arith::ConstantOp>(loc, tensor_type, weights_attr);

        result = constant_op.getResult();
        break;
    }
    case OperatorType::RELU:
    case OperatorType::MATMUL:
    case OperatorType::GEMM:
    case OperatorType::CONV:
    default:
        throw std::runtime_error("mlir conversion for this operation is not supported");
        break;
    }

    value_id_to_mlir_value[value] = result;
    return result;
};