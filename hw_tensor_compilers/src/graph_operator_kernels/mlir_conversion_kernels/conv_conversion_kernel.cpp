#include <unordered_map>
#include <memory>
#include <algorithm>
#include <iostream>

#include "graph/graph.h"
#include "passes/mlir_conversion_pass/mlir_conversion_data.h"
#include "passes/mlir_conversion_pass/utils.h"
#include "graph_operator_kernels/mlir_conversion_kernels/conv_conversion_kernel.h"

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace graph_engine;

#include "io/console_graph_exporter.h"



namespace passes::mlir_conversion {

    auto conv_pad(mlir::OpBuilder& builder, mlir::Location loc,
        mlir::Value input, const std::vector<int64_t>& pads) -> mlir::Value {
        // ONNX pads expected: [top, left, bottom, right]
        assert(pads.size() == 4 && "Expected 4 padding values");

        auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
        auto shape = input_type.getShape(); // {1, 1, 28, 28}

        llvm::SmallVector<int64_t, 4> new_shape;
        new_shape.push_back(shape[0]); // N
        new_shape.push_back(shape[1]); // C
        new_shape.push_back(shape[2] + pads[0] + pads[2]); // H + top + bottom
        new_shape.push_back(shape[3] + pads[1] + pads[3]); // W + left + right

        auto result_type = mlir::RankedTensorType::get(new_shape, input_type.getElementType());

        auto get_attr = [&](int64_t val) { return builder.getI64IntegerAttr(val); };

        auto getConstIndex = [&](int64_t val) -> mlir::OpFoldResult {
            return builder.getIndexAttr(val);
            };

        llvm::SmallVector<mlir::OpFoldResult> low_pads = {
            getConstIndex(0), getConstIndex(0), getConstIndex(pads[0]), getConstIndex(pads[1])
        };
        llvm::SmallVector<mlir::OpFoldResult> high_pads = {
            getConstIndex(0), getConstIndex(0), getConstIndex(pads[2]), getConstIndex(pads[3])
        };

        auto pad_op = builder.create<mlir::tensor::PadOp>(
            loc,
            result_type,
            input,
            /*low=*/low_pads,
            /*high=*/high_pads,
            /*nofold=*/false
        );

        {
            mlir::OpBuilder::InsertionGuard guard(builder);

            mlir::Region& region = pad_op.getRegion();

            auto input_type = input.getType().cast<mlir::RankedTensorType>();
            int64_t rank = input_type.getRank();
            llvm::SmallVector<mlir::Type, 4> block_arg_types(rank, builder.getIndexType());
            mlir::Block* block = builder.createBlock(
                &region, region.end(), block_arg_types,
                std::vector<mlir::Location>(rank, loc)
            );

            builder.setInsertionPointToStart(block);

            auto zero = builder.create<mlir::arith::ConstantOp>(
                loc, builder.getZeroAttr(input_type.getElementType()));
            builder.create<mlir::tensor::YieldOp>(loc, zero.getResult());
        }


        return pad_op.getResult();
    }

    auto make_zero_tensor(mlir::OpBuilder& builder, mlir::RankedTensorType type) -> mlir::Value {
        mlir::Location loc = builder.getUnknownLoc();

        auto empty_tensor = builder.create<mlir::tensor::EmptyOp>(
            loc,
            type.getShape(),
            type.getElementType()
        );

        mlir::Value zero;
        mlir::Type element_type = type.getElementType();

        if (mlir::isa<mlir::FloatType>(element_type)) {
            zero = builder.create<mlir::arith::ConstantFloatOp>(
                loc,
                llvm::APFloat(0.0),
                mlir::cast<mlir::FloatType>(element_type)
            );
        }
        else {
            zero = builder.create<mlir::arith::ConstantIntOp>(
                loc, 0, element_type
            );
        }

        auto fill_op = builder.create<mlir::linalg::FillOp>(
            loc,
            /*value=*/zero,
            /*output=*/empty_tensor.getResult()
        );

        mlir::Value zero_tensor = fill_op.getResult(0);

        return zero_tensor;
    }

    auto new_conv(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {
        // Given : mlir::Value input; mlir::Value filter
        // Dialects : linalg, arith, tensor
        // 
        // Expected output:
        // mlir::Value result = conv_op.getResult(0)

        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values.at(value_id).producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        //std::cout << "== input_init ==" << std::endl;
        //std::cout << graph.values.at(graph.nodes.at(producer_node).inputs[0]) << std::endl;
        mlir::Value input = storage.convert_graph_value(graph.nodes.at(producer_node).inputs[0]);
        //llvm::errs() << input;
        mlir::Value filter = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);
        //std::cout << "== end input_init ==" << std::endl;

        //std::cout << 0;
        
        // Attributes:
        // strides, dilations:  linalg::Conv2DOp takes attributes 
        // pads:                first apply padding, then linalg::Conv2DOp
        // kernel_shape:        mlir ignores and takes filter.shape
        // group:               requires different linalg functions or linalg::generic
        const auto& pads = std::get<std::vector<int64_t>>(graph.nodes.at(producer_node).attr.at("pads"));
        const auto& strides = std::get<std::vector<int64_t>>(graph.nodes.at(producer_node).attr.at("strides"));
        const auto& dilations = std::get<std::vector<int64_t>>(graph.nodes.at(producer_node).attr.at("dilations"));
        auto group = std::get<int64_t>(graph.nodes.at(producer_node).attr.at("group"));

        //std::cout << 0;
        
        // 0. Flatten shape to (NCHW)
        // if input tensor has rank > 4, it is flattened.
        // for example (n,t,c,h,w) -> (n*t,c,h,w) -> conv -> (n*t, c_out, h_out, w_out) -> (n,t,c_out, h_out, w_out)
        mlir::RankedTensorType output_type = get_value_tensor_type(builder, graph, value_id);
        mlir::Value init_tensor = builder.create<mlir::tensor::EmptyOp>(loc, output_type.getShape(), output_type.getElementType());
        mlir::Value output;
        mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(output_type.getElementType()));
        output = builder.create<mlir::linalg::FillOp>(loc, zero, init_tensor).result();
        /*
        // init destination with zeroes:
        if (graph.nodes[producer_node].inputs.size() < 3) {
            mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(output_type.getElementType()));
            output = builder.create<mlir::linalg::FillOp>(loc, zero, init_tensor).result();
        }
        else { // V3 as bias arg in Convolution:
            mlir::Value bias = storage.convert_graph_value(graph.nodes[producer_node].inputs[2]);
            auto bias_type = mlir::cast<mlir::RankedTensorType>(bias.getType());
            int64_t last_dim = output_type.getRank() - 1;

            // fill init_tensor with bias for all H, W
            output = builder.create<mlir::linalg::BroadcastOp>(
                loc,
                bias,
                init_tensor,
                mlir::ArrayRef<int64_t>{1} // broadcasted dims
            ).getResults()[0];
        }
        */
        auto input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
        if (input_type.getRank() != 4) throw std::runtime_error("MLIRConversion: Conv received input, which rank is not 4");

        // 1. Apply padding
        // padding is applied to dimensions H and W of input.
        bool pads_all_zeros = std::all_of(pads.begin(), pads.end(), [](int i) { return i == 0; });
        if (!pads_all_zeros) {
            //std::cout << " apply padding" << std::endl;
            //std::cout << "args: " << std::endl;
            //llvm::errs() << input;
            //std::cout << std::endl << "pads: ";
            //for (const auto& pad : pads) { std::cout << pad << " "; }
            input = conv_pad(builder, loc, input, pads);
            //llvm::errs() << input;
            //std::cout << " end apply padding ";
        }
        input_type = llvm::cast<mlir::RankedTensorType>(input.getType());
        


        //std::cout << 2;
        
        // 2. Convolution:
        // I: input.shape(batch_size, c_in, h_in, w_in)
        // W: filter.shape(c_out, c_in/group, k_h, k_w)
        // O: output.shape(batch_size, c_out, h_out, w_out)
        // s_h, s_w: strides
        // d_h, d_w: dilations

        auto filter_type = llvm::cast<mlir::RankedTensorType>(filter.getType());

        int64_t batch_size = output_type.getDimSize(0);
        int64_t c_out = output_type.getDimSize(1);
        int64_t h_out = output_type.getDimSize(2);
        int64_t w_out = output_type.getDimSize(3);
        int64_t k_h = filter_type.getDimSize(2);
        int64_t k_w = filter_type.getDimSize(3);
        int64_t d_h = dilations.at(0);
        int64_t d_w = dilations.at(1);
        int64_t s_h = strides.at(0);
        int64_t s_w = strides.at(1);

        // k in Range[0, c_in / group)  - reduction
        // m in Range[0, k_h)           - reduction
        // n in Range[0, k_w)           - reduction
        // 
        // b in Range[0, batch_size)    - parallel
        // c in Range[0, c_out)         - parallel
        // i in Range[0, h_out)         - parallel
        // j in Range[0, w_out)         - parallel
        //
        // g = floor( c / c_out * group)
        // output(b,c,i,j) = sum_{kmn}(
        //                              input[b, g*c_in/group + k, i*s_h + m*d_h, j*s_w + n*d_w] *
        //                              * filter[c_out, k, m, n]
        //                            )
        
        llvm::SmallVector<mlir::utils::IteratorType> iter_types = {
            mlir::utils::IteratorType::parallel,  // 0. b
            mlir::utils::IteratorType::parallel,  // 1. c
            mlir::utils::IteratorType::parallel,  // 2. i
            mlir::utils::IteratorType::parallel,  // 3. j
            mlir::utils::IteratorType::reduction, // 4. k
            mlir::utils::IteratorType::reduction, // 5. m
            mlir::utils::IteratorType::reduction  // 6. n
        };

        auto context = builder.getContext();
        mlir::AffineExpr b_expr = mlir::getAffineDimExpr(0, context); 
        mlir::AffineExpr c_expr = mlir::getAffineDimExpr(1, context);
        mlir::AffineExpr i_expr = mlir::getAffineDimExpr(2, context);
        mlir::AffineExpr j_expr = mlir::getAffineDimExpr(3, context);
        mlir::AffineExpr k_expr = mlir::getAffineDimExpr(4, context);
        mlir::AffineExpr m_expr = mlir::getAffineDimExpr(5, context);
        mlir::AffineExpr n_expr = mlir::getAffineDimExpr(6, context);

        //
        // g = floor( c / c_out * group)
        // output(b,c,i,j) = sum_{kmn}(
        //                              input[b, g*c_in/group + k, i*s_h + m*d_h, j*s_w + n*d_w] *
        //                              * filter[c_out, k, m, n]
        //                            )

        auto output_map = mlir::AffineMap::get(7, 0, { b_expr, c_expr, i_expr, j_expr }, context);
        auto filter_map = mlir::AffineMap::get(7, 0, { c_expr, k_expr, m_expr, n_expr }, context);

        // input indices:

        int64_t c_in_per_group = input_type.getDimSize(1) / group;
        int64_t c_out_per_group = c_out / group;
        
        mlir::AffineExpr input_c_index_expr;
        if (group == 1) {
            input_c_index_expr = k_expr;
        }
        else {
            auto g = c_expr.floorDiv(c_out_per_group);
            input_c_index_expr = g * c_in_per_group + k_expr;
        }
        auto input_h_index_expr = i_expr * s_h + m_expr * d_h;
        auto input_w_index_expr = j_expr * s_w + n_expr * d_w;

        auto input_map = mlir::AffineMap::get(7, 0, { b_expr, input_c_index_expr, input_h_index_expr, input_w_index_expr }, context);
        /*
        std::cout << 5;
        std::cout << std::endl << std::endl;

        std::cout << "== input: ==" << std::endl;
        llvm::errs() << input;
        std::cout << std::endl;

        std::cout << "== input_map: ==" << std::endl;
        llvm::errs() << input_map;
        std::cout << std::endl;

        std::cout << "== filter: ==" << std::endl;
        llvm::errs() << filter;
        std::cout << std::endl;

        std::cout << "== filter_map: ==" << std::endl;
        llvm::errs() << filter_map;
        std::cout << std::endl;

        std::cout << "== output: ==" << std::endl;
        llvm::errs() << output;
        std::cout << std::endl;

        std::cout << "== output_map: ==" << std::endl;
        llvm::errs() << output_map;
        std::cout << std::endl;

        std::cout << "== iter_types: ==" << std::endl;
        //llvm::errs() << iter_types;
        */


        llvm::SmallVector<mlir::AffineMap> maps = { input_map, filter_map, output_map };
        mlir::ValueRange inputs = { input, filter };
        mlir::ValueRange outputs = { output };

        auto conv_op = builder.create<mlir::linalg::GenericOp>(
            loc,
            /*resultTypes=*/mlir::TypeRange{ output_type },
            /*inputs=*/mlir::ValueRange{ input, filter },
            /*outputs=*/mlir::ValueRange{ output },
            /*indexingMaps=*/llvm::SmallVector<mlir::AffineMap>{ input_map, filter_map, output_map },
            /*iteratorTypes=*/iter_types,
            /*bodyBuild=*/[&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
                // blockArgs are: [input_val, filter_val, output_acc]
                mlir::Value mul = builder.create<mlir::arith::MulFOp>(loc, blockArgs[0], blockArgs[1]);
                mlir::Value add = builder.create<mlir::arith::AddFOp>(loc, blockArgs[2], mul);
                // acc + (input * filter)

                // new accumulator value
                builder.create<mlir::linalg::YieldOp>(loc, add);
            });

        // 3. Expand flattened dimensions
        // skip
        //std::cout << std::endl << 6 << " successful conv op" << std::endl;
        result = conv_op.getResult(0);
        //std::cout << 7;
        return result;
    }

	auto ConvConversionKernel::convert_graph_value(MLIRConversionData& storage, graph_engine::ValueID value_id) -> mlir::Value {

        return new_conv(storage, value_id);

        const Graph& graph = storage.graph;
        mlir::OpBuilder& builder = storage.builder;
        NodeID producer_node = graph.values[value_id].producer_node_id;
        mlir::Location loc = mlir::FileLineColLoc::get(builder.getContext(), "graph", producer_node, value_id);

        mlir::Value result;

        mlir::Value input = storage.convert_graph_value(graph.nodes[producer_node].inputs[0]);
        mlir::Value filter = storage.convert_graph_value(graph.nodes[producer_node].inputs[1]);

        auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
        auto filter_type = mlir::cast<mlir::RankedTensorType>(filter.getType());
        auto element_type = input_type.getElementType();
        auto output_type = mlir_conversion::get_value_tensor_type(builder, graph, value_id);

        mlir::Value init_tensor = builder.create<mlir::tensor::EmptyOp>(loc, output_type.getShape(), output_type.getElementType());
        mlir::Value dest;

        // init destination with zeroes:
        if (graph.nodes[producer_node].inputs.size() < 3) {
            mlir::Value zero = builder.create<mlir::arith::ConstantOp>(loc, builder.getZeroAttr(element_type));
            dest = builder.create<mlir::linalg::FillOp>(loc, zero, init_tensor).result();
        }
        else { // V3 as bias arg in Convolution:
            mlir::Value bias = storage.convert_graph_value(graph.nodes[producer_node].inputs[2]);
            auto bias_type = mlir::cast<mlir::RankedTensorType>(bias.getType());
            int64_t last_dim = output_type.getRank() - 1;

            // fill init_tensor with bias for all H, W
            dest = builder.create<mlir::linalg::BroadcastOp>(
                loc,
                bias,
                init_tensor,
                mlir::ArrayRef<int64_t>{0, 2, 3} // broadcasted dims
            ).getResults()[0];
        }

        // Affine Maps :
        // Iterators and Conv indices: d0=N, d1=H, d2=W, d3=F, d4=KH, d5=KW, d6=C
        auto map_input = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(0),
                                                     builder.getAffineDimExpr(1) + builder.getAffineDimExpr(4),
                                                     builder.getAffineDimExpr(2) + builder.getAffineDimExpr(5),
                                                     builder.getAffineDimExpr(6) }, builder.getContext());
        auto map_filter = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(4),
                                                      builder.getAffineDimExpr(5),
                                                      builder.getAffineDimExpr(6),
                                                      builder.getAffineDimExpr(3) }, builder.getContext());
        auto map_output = mlir::AffineMap::get(7, 0, { builder.getAffineDimExpr(0),
                                                      builder.getAffineDimExpr(1),
                                                      builder.getAffineDimExpr(2),
                                                      builder.getAffineDimExpr(3) }, builder.getContext());

        // Iterators:
        llvm::SmallVector<mlir::AffineMap, 3> indexing_maps = { map_input, map_filter, map_output };
        llvm::SmallVector<mlir::utils::IteratorType> iterTypes(4, mlir::utils::IteratorType::parallel);
        iterTypes.append(3, mlir::utils::IteratorType::reduction);
        auto convOp = builder.create<mlir::linalg::GenericOp>(
            loc,
            output_type,
            mlir::ValueRange{ input, filter }, // V0, V2
            mlir::ValueRange{ dest },          // V1 (output buffer)
            indexing_maps,
            iterTypes,
            [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange args) {
                // args[0] - input, args[1] - filter, args[2] - output_acc
                mlir::Value mul = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                mlir::Value add = builder.create<mlir::arith::AddFOp>(loc, mul, args[2]);
                builder.create<mlir::linalg::YieldOp>(loc, add);
            });
        result = convOp.getResult(0);

        return result;
	}
}