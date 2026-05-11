#pragma once
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/graph.h"
#include "io/onnx_importer.h"
#include "passes/passes.h"

#include "mlir/IR/MLIRContext.h"



TEST(MLIRPipeline, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, SingleAddModel) {
    std::string filename = "data/single_add.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, SingleReluModel) {
    std::string filename = "data/single_relu.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, SingleMatMulModel) {
    std::string filename = "data/single_matmul.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, SingleGemmModel) {
    std::string filename = "data/single_gemm.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};

TEST(MLIRPipeline, TinyModel) {
    std::string filename = "data/tiny.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::mlir_management::set_context(context);
    passes::MLIRConversionPass conversion_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> model = conversion_pass.convert();
    EXPECT_TRUE(mlir::succeeded(model->verify()));

    mlir::LogicalResult result = passes::mlir_management::lower_to_llvm(*model, false);
    EXPECT_TRUE(mlir::succeeded(result));
};