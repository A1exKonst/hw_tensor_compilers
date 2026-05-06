#pragma once
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "graph/graph.h"
#include "io/onnx_importer.h"
#include "io/console_graph_exporter.h"

TEST(OnnxImport, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 3);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(2));

    graph_engine::Node& node = graph.nodes[0];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::MUL);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(2));

    graph_engine::Value& v1 = graph.values[0];
    graph_engine::Value& v2 = graph.values[1];
    graph_engine::Value& out = graph.values[2];

    EXPECT_THAT(v1.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(v2.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(out.consumer_node_ids, ::testing::ElementsAre());


    EXPECT_EQ(v1.producer_node_id, size_t(-1));
    EXPECT_EQ(v2.producer_node_id, size_t(-1));
    EXPECT_EQ(out.producer_node_id, 0);

    EXPECT_EQ(v1.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(v2.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(out.dtype, graph_engine::DataType::FLOAT32);

    EXPECT_THAT(v1.shape, ::testing::ElementsAre(1, 10));
    EXPECT_THAT(v2.shape, ::testing::ElementsAre(1, 10));
    EXPECT_THAT(out.shape, ::testing::ElementsAre(1, 10));
};

TEST(OnnxImport, SingleAddModel) {
    std::string filename = "data/single_add.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 3);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(2));

    graph_engine::Node& node = graph.nodes[0];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::ADD);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(2));

    graph_engine::Value& v1 = graph.values[0];
    graph_engine::Value& v2 = graph.values[1];
    graph_engine::Value& out = graph.values[2];

    EXPECT_THAT(v1.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(v2.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(out.consumer_node_ids, ::testing::ElementsAre());


    EXPECT_EQ(v1.producer_node_id, size_t(-1));
    EXPECT_EQ(v2.producer_node_id, size_t(-1));
    EXPECT_EQ(out.producer_node_id, 0);

    EXPECT_EQ(v1.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(v2.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(out.dtype, graph_engine::DataType::FLOAT32);

    EXPECT_THAT(v1.shape, ::testing::ElementsAre(1, 10));
    EXPECT_THAT(v2.shape, ::testing::ElementsAre(1, 10));
    EXPECT_THAT(out.shape, ::testing::ElementsAre(1, 10));
};

TEST(OnnxImport, SingleReluModel) {
    std::string filename = "data/single_relu.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 2);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(1));

    graph_engine::Node& node = graph.nodes[0];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::RELU);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(1));

    graph_engine::Value& v1 = graph.values[0];
    graph_engine::Value& out = graph.values[1];

    EXPECT_THAT(v1.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(out.consumer_node_ids, ::testing::ElementsAre());

    EXPECT_EQ(v1.producer_node_id, size_t(-1));
    EXPECT_EQ(out.producer_node_id, 0);

    EXPECT_EQ(v1.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(out.dtype, graph_engine::DataType::FLOAT32);

    EXPECT_THAT(v1.shape, ::testing::ElementsAre(1, 10));
    EXPECT_THAT(out.shape, ::testing::ElementsAre(1, 10));
};

TEST(OnnxImport, SingleMatMulModel) {
    std::string filename = "data/single_matmul.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 3);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(2));

    graph_engine::Node& node = graph.nodes[0];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::MATMUL);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(2));

    graph_engine::Value& v1 = graph.values[0];
    graph_engine::Value& v2 = graph.values[1];
    graph_engine::Value& out = graph.values[2];

    EXPECT_THAT(v1.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(v2.consumer_node_ids, ::testing::ElementsAre(0));
    EXPECT_THAT(out.consumer_node_ids, ::testing::ElementsAre());

    EXPECT_EQ(v1.producer_node_id, size_t(-1));
    EXPECT_EQ(v2.producer_node_id, size_t(-1));
    EXPECT_EQ(out.producer_node_id, 0);

    EXPECT_EQ(v1.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(v2.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(out.dtype, graph_engine::DataType::FLOAT32);

    EXPECT_THAT(v1.shape, ::testing::ElementsAre(5, 10));
    EXPECT_THAT(v2.shape, ::testing::ElementsAre(10, 5));
    EXPECT_THAT(out.shape, ::testing::ElementsAre(5, 5));
};

TEST(OnnxImport, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 3);
    ASSERT_EQ(graph.values.size(), 4);

    EXPECT_THAT(graph.inputs,   ::testing::ElementsAre(0));
    EXPECT_THAT(graph.outputs,  ::testing::ElementsAre(1));

    // Nodes:

    for (int i = 0; i < 2; ++i) {
        EXPECT_EQ(graph.nodes[i].op_type, graph_engine::OperatorType::CONSTANT);
        EXPECT_THAT(graph.nodes[i].inputs, ::testing::ElementsAre());
        EXPECT_THAT(graph.nodes[i].outputs, ::testing::ElementsAre((2 + i)));
    }
    graph_engine::Node& node = graph.nodes[2];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::CONV);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0, 2, 3));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(1));

    // Nodes Conv Attrs:
    EXPECT_EQ(std::get<int64_t>(node.attr.at("group")),                         1);
    EXPECT_THAT(std::get<std::vector<int64_t>>(node.attr.at("dilations")),      ::testing::ElementsAre(1,1));
    EXPECT_THAT(std::get<std::vector<int64_t>>(node.attr.at("kernel_shape")),   ::testing::ElementsAre(3,3));
    EXPECT_THAT(std::get<std::vector<int64_t>>(node.attr.at("pads")),           ::testing::ElementsAre(1,1,1,1));
    EXPECT_THAT(std::get<std::vector<int64_t>>(node.attr.at("strides")),        ::testing::ElementsAre(1,1));

    // Values:

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(graph.values[i].dtype, graph_engine::DataType::FLOAT32);
    }

    EXPECT_EQ(graph.values[0].producer_node_id, size_t(-1));
    EXPECT_EQ(graph.values[1].producer_node_id, 2);
    EXPECT_EQ(graph.values[2].producer_node_id, 0);
    EXPECT_EQ(graph.values[3].producer_node_id, 1);

    EXPECT_THAT(graph.values[0].consumer_node_ids, ::testing::ElementsAre(2));
    EXPECT_THAT(graph.values[1].consumer_node_ids, ::testing::ElementsAre());
    EXPECT_THAT(graph.values[2].consumer_node_ids, ::testing::ElementsAre(2));
    EXPECT_THAT(graph.values[3].consumer_node_ids, ::testing::ElementsAre(2));

    EXPECT_THAT(graph.values[0].shape, ::testing::ElementsAre(1, 1, 28, 28));
    EXPECT_THAT(graph.values[1].shape, ::testing::ElementsAre(1, 1, 28, 28));
    EXPECT_THAT(graph.values[2].shape, ::testing::ElementsAre(1, 1, 3, 3));
    EXPECT_THAT(graph.values[3].shape, ::testing::ElementsAre(1));
};

TEST(OnnxImport, SingleGemmModel) {
    std::string filename = "data/single_gemm.onnx";
    graph_engine::Graph graph = io::OnnxImporter(filename).import_graph();

    ASSERT_EQ(graph.nodes.size(), 3);
    ASSERT_EQ(graph.values.size(), 4);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(1));

    graph_engine::Node& node_const1 = graph.nodes[0];
    EXPECT_EQ(node_const1.op_type, graph_engine::OperatorType::CONSTANT);
    EXPECT_THAT(node_const1.inputs, ::testing::ElementsAre());
    EXPECT_THAT(node_const1.outputs, ::testing::ElementsAre(2));

    graph_engine::Node& node_const2 = graph.nodes[1];
    EXPECT_EQ(node_const2.op_type, graph_engine::OperatorType::CONSTANT);
    EXPECT_THAT(node_const2.inputs, ::testing::ElementsAre());
    EXPECT_THAT(node_const2.outputs, ::testing::ElementsAre(3));

    graph_engine::Node& node = graph.nodes[2];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::GEMM);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0, 2, 3));
    EXPECT_THAT(node.outputs, ::testing::ElementsAre(1));

    graph_engine::Value& in = graph.values[0];
    graph_engine::Value& out = graph.values[1];
    graph_engine::Value& v2 = graph.values[2];
    graph_engine::Value& v3 = graph.values[3];

    EXPECT_THAT(in.consumer_node_ids, ::testing::ElementsAre(2));
    EXPECT_THAT(v2.consumer_node_ids, ::testing::ElementsAre(2));
    EXPECT_THAT(v2.consumer_node_ids, ::testing::ElementsAre(2));
    EXPECT_THAT(out.consumer_node_ids, ::testing::ElementsAre());

    EXPECT_EQ(in.producer_node_id, size_t(-1));
    EXPECT_EQ(v2.producer_node_id, 0);
    EXPECT_EQ(v3.producer_node_id, 1);
    EXPECT_EQ(out.producer_node_id, 2);

    EXPECT_EQ(in.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(v2.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(v3.dtype, graph_engine::DataType::FLOAT32);
    EXPECT_EQ(out.dtype, graph_engine::DataType::FLOAT32);

    bool is_in_value_correct = (in.shape.rank() == 2) && (in.shape[0] == 1) && (in.shape[1] == 10);
    bool is_value2_correct = (v2.shape.rank() == 2) && (v2.shape[0] == 20) && (v2.shape[1] == 10);
    bool is_value3_correct = (v3.shape.rank() == 1) && (v3.shape[0] == 20);
    bool is_output_value_correct = (out.shape.rank() == 2) && (out.shape[0] == 1) && (out.shape[1] == 20);

    EXPECT_TRUE(is_in_value_correct);
    EXPECT_TRUE(is_value2_correct);
    EXPECT_TRUE(is_value3_correct);
    EXPECT_TRUE(is_output_value_correct);

    // Gemm atributes:
    ASSERT_TRUE(node.attr.find("alpha") != node.attr.end());
    EXPECT_EQ(std::get<float>(node.attr.at("alpha")), 1);
    ASSERT_TRUE(node.attr.find("transB") != node.attr.end());
    EXPECT_EQ((bool)std::get<int64_t>(node.attr.at("transB")), true);
    ASSERT_TRUE(node.attr.find("beta") != node.attr.end());
    EXPECT_EQ(std::get<float>(node.attr.at("beta")), 1);
};