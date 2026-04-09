#pragma once
#include <stdexcept>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/graph.h"
#include "io/in_graph_onnx.h"
#include "passes/semantics_inferer.h"

TEST(Semantics, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);
    
    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};

TEST(Semantics, SingleAddModel) {
    std::string filename = "data/single_add.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};

TEST(Semantics, SingleReluModel) {
    std::string filename = "data/single_relu.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};

TEST(Semantics, SingleMatMulModel) {
    std::string filename = "data/single_matmul.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};

TEST(Semantics, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};

TEST(Semantics, SingleGemmModelInvalidAddition) {
    std::string filename = "data/single_gemm.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    EXPECT_NO_THROW(passes::SemanticsInferer::transform_graph(graph));
};