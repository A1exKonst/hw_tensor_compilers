#pragma once
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "graph/graph.h"
#include "io/in_graph_onnx.h"
#include "io/out_graph_console.h"


TEST(GraphShape, ShapeInitialization) {
    graph_engine::Shape shape = graph_engine::Shape();
    ASSERT_EQ(shape.rank(), 0) << "ERR: rank after empty initialization is not 0";
}

TEST(GraphShape, ShapeRankChange) {
    graph_engine::Shape shape = graph_engine::Shape();
    shape.rank(2);
    shape[0] = 5;
    shape[1] = 9;
    shape.rank(1);
    EXPECT_EQ(shape[0], 5) << "ERR: shape rank change cannot affect lower ranks";
    shape.rank(3);
    EXPECT_EQ(shape[1], 1) << "ERR: shape rank change: deleted rank dims was not cleared";
    EXPECT_EQ(shape[2], 1) << "ERR: new shape dims was not initialized with '1'";
};

TEST(GraphShape, ShapeBroadcasting_1) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 11; s1[1] = 12; s1[2] = 13;
    s2[0] = 11; s2[1] = 12; s2[2] = 13;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(11, 12, 13));
};

TEST(GraphShape, ShapeBroadcasting_2) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 11; s1[1] = 12; s1[2] = 13;
    s2[0] = 11; s2[1] = 14; s2[2] = 13;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);
    EXPECT_FALSE(result.has_value());
};

TEST(GraphShape, ShapeBroadcasting_3) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 11; s1[1] = 12; s1[2] = 1;
    s2[0] = 11; s2[1] = 12; s2[2] = 13;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(11, 12, 13));
};

TEST(GraphShape, ShapeBroadcasting_4) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(5);
    std::optional<graph_engine::Shape> result;

    s1[0] = 11; s1[1] = 12; s1[2] = 1;
    s2[0] = 11; s2[1] = 12; s2[2] = 13;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_FALSE(result.has_value());
};

TEST(GraphShape, ShapeBroadcasting_5) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(2);
    std::optional<graph_engine::Shape> result;

    s1[0] = 8; s1[1] = 3; s1[2] = 50; s1[3] = 50;
    s2[0] = 50; s2[1] = 1;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(8,3,50,50));
};

TEST(GraphShape, ShapeBroadcasting_6) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(1);
    std::optional<graph_engine::Shape> result;

    s1[0] = 5; s1[1] = 1; s1[2] = 1;
    s2[0] = 10;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(5,1,10));
};

TEST(OnnxImport, SingleMulModel) {
    std::string filename = "data/single_mul.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 3);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0,1));
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
    graph_engine::Graph graph = io::import_from_model(filename);

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
    graph_engine::Graph graph = io::import_from_model(filename);

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
    graph_engine::Graph graph = io::import_from_model(filename);

    ASSERT_EQ(graph.nodes.size(), 1);
    ASSERT_EQ(graph.values.size(), 3);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0, 1));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(2));

    graph_engine::Node& node = graph.nodes[0];
    EXPECT_EQ(node.op_type, graph_engine::OperatorType::MATMUL);
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0,1));
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

    EXPECT_THAT(v1.shape, ::testing::ElementsAre(5,10));
    EXPECT_THAT(v2.shape, ::testing::ElementsAre(10, 5));
    EXPECT_THAT(out.shape, ::testing::ElementsAre(5, 5));
};

TEST(OnnxImport, SingleConvModel) {
    std::string filename = "data/single_conv.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

    ASSERT_EQ(graph.nodes.size(), 3);
    ASSERT_EQ(graph.values.size(), 4);

    EXPECT_THAT(graph.inputs, ::testing::ElementsAre(0));
    EXPECT_THAT(graph.outputs, ::testing::ElementsAre(1));

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

    // TODO: Node Conv Attrs

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

    EXPECT_THAT(graph.values[0].shape, ::testing::ElementsAre(1,1,28,28));
    EXPECT_THAT(graph.values[1].shape, ::testing::ElementsAre(1, 1, 28, 28));
    EXPECT_THAT(graph.values[2].shape, ::testing::ElementsAre(1, 1, 3, 3));
    EXPECT_THAT(graph.values[3].shape, ::testing::ElementsAre(1));

    ASSERT_TRUE(false) << " OnnxImport::SingleConvModel: test not ready to be used";
};

TEST(OnnxImport, SingleGemmModel) {
    std::string filename = "data/single_gemm.onnx";
    graph_engine::Graph graph = io::import_from_model(filename);

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
    EXPECT_THAT(node.inputs, ::testing::ElementsAre(0,2,3));
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
    EXPECT_EQ((bool) std::get<int64_t>(node.attr.at("transB")), true);
    ASSERT_TRUE(node.attr.find("beta") != node.attr.end());
    EXPECT_EQ(std::get<float>(node.attr.at("beta")), 1);
};