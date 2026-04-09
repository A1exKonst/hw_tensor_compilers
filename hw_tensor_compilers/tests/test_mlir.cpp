#pragma once
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/graph.h"
#include "io/in_graph_onnx.h"
#include "passes/semantics_inferer.h"
#include "passes/mlir_converter.h"

#include "mlir/IR/MLIRContext.h"



TEST(MLIRConversion, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};

TEST(MLIRConversion, SingleAddModel) {
    std::string filename = "data/single_add.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};

TEST(MLIRConversion, SingleReluModel) {
    std::string filename = "data/single_relu.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};

TEST(MLIRConversion, SingleMatMulModel) {
    std::string filename = "data/single_matmul.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};

TEST(MLIRConversion, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};

TEST(MLIRConversion, SingleGemmModel) {
    std::string filename = "data/single_gemm.onnx";

    graph_engine::Graph graph = io::import_from_model(filename);
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));

    mlir::MLIRContext context;
    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));
};