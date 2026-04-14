#include "passes/mlir_converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace graph_engine;
using namespace passes;
using LinalgRegionBuilder = std::function<void(mlir::OpBuilder&, mlir::Location, mlir::ValueRange)>;





auto GraphToMLIRConverter::convert() -> mlir::OwningOpRef<mlir::ModuleOp> {
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

    auto loc = builder.getUnknownLoc();

    mlir::ModuleOp module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    mlir::FunctionType func_type = get_function_type(builder, graph);
    mlir::func::FuncOp func_op = builder.create<mlir::func::FuncOp>(loc, "main", func_type);

    mlir::Block* entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // === void convert_graph_nodes() :

    // todo: add graph.nodes visit
    // via convert_value_to_mlir_value

    if (entry_block->getNumArguments() != graph.inputs.size()) {
        throw std::runtime_error("Conversion to MLIR: Wrong number of inputs in entry_block");
    }

    for (size_t i = 0; i < entry_block->getNumArguments(); ++i) {
        value_id_to_mlir_value[graph.inputs[i]] = entry_block->getArgument(i);
    }

    for (ValueID output : graph.outputs) {
        convert_graph_value_to_mlir_recursively(output);
    }

    std::vector<mlir::Value> return_values;
    for (ValueID output : graph.outputs) {
        return_values.push_back(value_id_to_mlir_value[output]);
    }
    builder.create<mlir::func::ReturnOp>(loc, return_values);

    return mlir::OwningOpRef<mlir::ModuleOp>(module);
};

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
        throw std::runtime_error("Conversion to MLIR: Invalid graph_engine::DataType encountered");
    }
    return return_type;
}

auto GraphToMLIRConverter::get_value_tensor_type(mlir::OpBuilder& builder, const graph_engine::Graph& graph, ValueID value_id) -> mlir::RankedTensorType {
    mlir::Type dtype = datatype_to_mlir_type(builder, graph.values[value_id].dtype);
    const Shape& s = graph.values[value_id].shape;
    llvm::SmallVector<int64_t, graph_engine::MAX_VALUE_RANK> shape_slice(s.begin(), s.end());
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
    mlir::Value lhs, mlir::Value rhs,
    mlir::OpBuilder& builder, mlir::Location loc) {
    if (lhs.getType().isa<mlir::FloatType>()) {
        return builder.create<FloatOp>(loc, lhs, rhs).getResult();
    }
    return builder.create<IntOp>(loc, lhs, rhs).getResult();
}

template <typename IntOp, typename FloatOp>
mlir::Value GraphToMLIRConverter::create_binary_operation(NodeID producer) {
    mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer, 0);
    mlir::Value lhs = convert_graph_value_to_mlir_recursively(graph.nodes[producer].inputs[0]);
    mlir::Value rhs = convert_graph_value_to_mlir_recursively(graph.nodes[producer].inputs[1]);

    return create_mlir_binary_operation<IntOp, FloatOp>(lhs, rhs, builder, loc);
};

auto GraphToMLIRConverter::convert_graph_value_to_mlir_recursively(graph_engine::ValueID value) -> mlir::Value {

    // check if already done when recurringly call conversion
    if (value_id_to_mlir_value.find(value) != value_id_to_mlir_value.end()) {
        return value_id_to_mlir_value.at(value);
    }

    mlir::Value result;

    NodeID producer_node = graph.values[value].producer_node_id;

    mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, 0);

    // ==== get result depending on operation:

    OperatorType op_type = graph.nodes[producer_node].op_type;

    switch (op_type) {
    case OperatorType::ADD:{
        result = create_binary_operation<mlir::arith::AddIOp, mlir::arith::AddFOp>(producer_node);
        break;
    }
    case OperatorType::MUL: {
        result = create_binary_operation<mlir::arith::MulIOp, mlir::arith::MulFOp>(producer_node);
        break;
    }
    case OperatorType::CONSTANT:{
        auto& weights = std::get<std::vector<float>>(graph.nodes[producer_node].attr.at("weights"));
        auto tensor_type = get_value_tensor_type(builder, graph, value);
        auto weights_attr = mlir::DenseElementsAttr::get(tensor_type, llvm::ArrayRef(weights));
        auto constant_op = builder.create<mlir::arith::ConstantOp>(loc, tensor_type, weights_attr);
        result = constant_op.getResult();
        break;
    }
    case OperatorType::RELU: {
        mlir::Value input = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[0]);
        mlir::Type elementType = mlir::cast<mlir::ShapedType>(input.getType()).getElementType();

        // ==== create zero tensor (for comparison in relu):

        mlir::TypedAttr zero_attr;
        if (elementType.isa<mlir::FloatType>()) {
            zero_attr = builder.getFloatAttr(elementType, 0.0);
        }
        else {
            zero_attr = builder.getIntegerAttr(elementType, 0);
        }
        mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, zero_attr);

        mlir::Value output = builder.create<mlir::tensor::EmptyOp>(
            loc, mlir::cast<mlir::RankedTensorType>(input.getType()).getShape(), elementType);

        // ==== create elementwise operation map:
        int64_t rank = mlir::cast<mlir::RankedTensorType>(input.getType()).getRank();
        mlir::AffineMap map = builder.getMultiDimIdentityMap(rank); // elementwise operation -> maps are identical
        llvm::SmallVector<mlir::AffineMap> maps = { map, map };     // 2 maps : for input + output

        // ==== create elementwise iterators:
        // an iterator is required for each dim, {rank} iterators total
        // element operations are independent -> iterators are parallel
        llvm::SmallVector<mlir::utils::IteratorType> iterators(rank, mlir::utils::IteratorType::parallel);

        // ==== choose arith::Max operation:
        LinalgRegionBuilder lambda_arith_max;
        if (elementType.isa<mlir::FloatType>()) {
            lambda_arith_max = [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange args) {
                auto max = b.create<mlir::arith::MaximumFOp>(l, args[0], zero);  // args[0] - input element
                b.create<mlir::linalg::YieldOp>(l, max.getResult());             // args[1] - output element
                };
        }
        else {
            lambda_arith_max = [&](mlir::OpBuilder& b, mlir::Location l, mlir::ValueRange args) {
                auto max = b.create<mlir::arith::MaxSIOp>(l, args[0], zero);
                b.create<mlir::linalg::YieldOp>(l, max.getResult());
                };
        }

        // ==== create RELU (linalg.generic)
        mlir::linalg::GenericOp relu_op;
        relu_op = builder.create<mlir::linalg::GenericOp>(
            loc,                // location 
            input.getType(),    // result type
            input,              // relu input
            output,             // relu output
            maps,               // elementwise operation maps
            iterators,          // elementwise operation iterators
            lambda_arith_max);  // operation

        result = relu_op.getResult(0);
        break;
    }
    case OperatorType::GEMM: {
        // Operation: result = alpha*A @ B + C;

        mlir::Value input_A = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[0]);
        mlir::Value input_B = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[1]);
        mlir::Value input_C = convert_graph_value_to_mlir_recursively(graph.nodes[producer_node].inputs[2]);

        bool transB = bool(std::get<int64_t>(graph.nodes[producer_node].attr.at("transB")));
        float alpha = std::get<float>(graph.nodes[producer_node].attr.at("alpha"));
        float beta = std::get<float>(graph.nodes[producer_node].attr.at("beta"));

        mlir::Value matmul_result = matmul(input_A, input_B, builder, loc, transB);
        mlir::Value alpha_result = scalar_mul(matmul_result, alpha, builder, loc);
        mlir::Value beta_result = scalar_mul(input_C, beta, builder, loc);

        // elementwise add:
        result = create_mlir_binary_operation<mlir::arith::AddIOp, mlir::arith::AddFOp>(alpha_result, beta_result, builder, loc);
        break;
    }
    case OperatorType::MATMUL:
    case OperatorType::CONV:
    default:
        throw std::runtime_error(
            "mlir conversion for this operation is not supported: " + 
            operator_type_to_str.at(op_type));
        break;
    }

    // assert result type is same, as given in graph_engine::Value::dtype :
    if (result.getType() != get_value_tensor_type(builder, graph, value)) { throw std::runtime_error("Conversion to MLIR: expected another mlir::Type for graph_engine::Value " + std::to_string(value)); };

    value_id_to_mlir_value[value] = result;
    return result;
};

auto passes::matmul(mlir::Value a, mlir::Value b, mlir::OpBuilder& builder, mlir::Location loc, bool transpose_b) -> mlir::Value {

    int64_t M = a.getType().cast<mlir::RankedTensorType>().getDimSize(0);       // A is (M x K)
    unsigned short N_index = (unsigned short)(!transpose_b);
    int64_t N = b.getType().cast<mlir::RankedTensorType>().getDimSize(N_index); // B is (K x N)

    mlir::Type elementType = a.getType().cast<mlir::RankedTensorType>().getElementType();

    // Empty tensor for matmul typing : RankedTensor (M x N)
    mlir::Value matmul_init = builder.create<mlir::tensor::EmptyOp>(
        loc,
        mlir::ArrayRef<int64_t>{M, N},
        elementType
    );
    
    mlir::Operation* matmul_op;
    if (transpose_b) {
        matmul_op = builder.create<mlir::linalg::MatmulTransposeBOp>(
            loc,
            mlir::TypeRange{ matmul_init.getType() },   //result TensorType
            mlir::ValueRange{ a, b },                   //inputs
            mlir::ValueRange{ matmul_init }             //outputs
        );
    }
    else {
        matmul_op = builder.create<mlir::linalg::MatmulOp>(
            loc,
            mlir::TypeRange{ matmul_init.getType() },   //result TensorType
            mlir::ValueRange{ a, b },                   //inputs
            mlir::ValueRange{ matmul_init }             //outputs
        );
    };

    // because SSA, matmul_init is a start state of output of operation (zeroed tensor), 
    // and is not a result of matmul
    mlir::Value matmul_result = matmul_op->getResult(0);
    return matmul_result;
};

auto passes::scalar_mul(mlir::Value A, float s, mlir::OpBuilder& builder, mlir::Location loc) -> mlir::Value {
    mlir::Type type = A.getType().cast<mlir::ShapedType>().getElementType();

    mlir::Value scalar = builder.create<mlir::arith::ConstantFloatOp>(
        loc, llvm::APFloat(s), type.cast<mlir::FloatType>());

    mlir::arith::MulFOp op = builder.create<mlir::arith::MulFOp>(loc, A, scalar);

    return op.getResult();
}