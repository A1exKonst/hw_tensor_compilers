#pragma once
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/graph.h"
#include "io/onnx_importer.h"
#include "io/console_graph_exporter.h"
#include "passes/passes.h"
#include "passes/mlir_conversion_pass/mlir_conversion_pass.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"


TEST(MLIRConversionPassIdentity, SingleAddModel) {
    std::string filename = "data/single_add.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));
    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    model->dump();

    std::cout << "========================= new model ===========================";

    pass_model->dump();

    EXPECT_TRUE(identical);
}

TEST(MLIRConversionPassIdentity, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    EXPECT_TRUE(identical);
}

TEST(MLIRConversionPassIdentity, SingleGemmModel) {
    std::string filename = "data/single_gemm.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    EXPECT_TRUE(identical);
}

TEST(MLIRConversionPassIdentity, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    EXPECT_TRUE(identical);
}

TEST(MLIRConversionPassIdentity, SingleReluModel) {
    std::string filename = "data/single_relu.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    EXPECT_TRUE(identical);
}

TEST(MLIRConversionPassIdentity, TinyModel) {
    std::string filename = "data/tiny.onnx";

    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();
    EXPECT_NO_THROW(passes::SemanticsInfererPass().transform_graph(graph));

    mlir::MLIRContext context;
    passes::llvm_mlir_management::set_context(context);

    passes::GraphToMLIRConverter converter{ context, graph };
    mlir::OwningOpRef<mlir::ModuleOp> model = converter.convert();

    passes::MLIRConversionPass mlir_pass(graph, context);
    mlir::OwningOpRef<mlir::ModuleOp> pass_model = mlir_pass.convert();

    bool identical = mlir::OperationEquivalence::isEquivalentTo(
        model->getOperation(),
        pass_model->getOperation(),
        mlir::OperationEquivalence::Flags::None //mlir::OperationEquivalence::Flags::IgnoreLocations
    );

    EXPECT_TRUE(identical);
}