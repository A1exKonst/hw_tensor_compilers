#pragma once
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "graph/shape.h"

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

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(11,12,13,1,1));
};

TEST(GraphShape, ShapeBroadcasting_5) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(2);
    std::optional<graph_engine::Shape> result;

    s1[0] = 50; s1[1] = 50; s1[2] = 3; s1[3] = 8;
    s2[0] = 50; s2[1] = 1;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(50, 50, 3, 8));
};

TEST(GraphShape, ShapeBroadcasting_6) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(1);
    std::optional<graph_engine::Shape> result;

    s1[0] = 1; s1[1] = 1; s1[2] = 5;
    s2[0] = 10;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(10, 1, 5));
};

TEST(GraphShape, ShapeBroadcasting_7) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(1);
    std::optional<graph_engine::Shape> result;

    s1[0] = 3; s1[1] = 1920; s1[2] = 1080; s1[3] = 1;
    s2[0] = 3;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(3, 1920, 1080, 1));
};

TEST(GraphShape, ShapeBroadcasting_8) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(2);
    std::optional<graph_engine::Shape> result;

    s1[0] = 4; s1[1] = 5; s1[2] = 10;
    s2[0] = 4; s2[1] = 1;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(4, 5, 10));
};

TEST(GraphShape, ShapeBroadcasting_9) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 64; s1[1] = 64; s1[2] = 1; s1[3] = 8;
    s2[0] = 1; s2[1] = 1; s2[2] = 3;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(64,64,3,8));
};

TEST(GraphShape, ShapePartialBroadcasting_1) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 64; s1[1] = 64; s1[2] = 1; s1[3] = 8;
    s2[0] = 1; s2[1] = 1; s2[2] = 3;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2, 1);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(1, 64, 3, 8));
};

TEST(GraphShape, ShapePartialBroadcasting_2) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 64; s1[1] = 64; s1[2] = 1; s1[3] = 8;
    s2[0] = 1; s2[1] = 1; s2[2] = 3;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2, 2);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(1, 1, 3, 8));
};

TEST(GraphShape, ShapePartialBroadcasting_3) {
    graph_engine::Shape s1 = graph_engine::Shape(4);
    graph_engine::Shape s2 = graph_engine::Shape(3);
    std::optional<graph_engine::Shape> result;

    s1[0] = 64; s1[1] = 64; s1[2] = 1; s1[3] = 8;
    s2[0] = 1; s2[1] = 1; s2[2] = 3;
    result = graph_engine::calculate_broadcast_compatible_shape(s1, s2, 3);

    EXPECT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(1, 1, 1, 8));
};

TEST(GraphShape, ShapeMatmul_1) {
    graph_engine::Shape s1 = graph_engine::Shape(2);
    graph_engine::Shape s2 = graph_engine::Shape(2);
    std::optional<graph_engine::Shape> result;

    s1[0] = 4; s1[1] = 5;
    s2[0] = 2; s2[1] = 4;
    result = graph_engine::calculate_matmul_compatible_shape(s1, s2);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(2,5));
}

TEST(GraphShape, ShapeMatmul_2) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(2);
    std::optional<graph_engine::Shape> result;

    s1[0] = 4; s1[1] = 5; s1[2] = 10;
    s2[0] = 2; s2[1] = 4;
    result = graph_engine::calculate_matmul_compatible_shape(s1, s2);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(2, 5, 10));
}

TEST(GraphShape, ShapeMatmul_3) {
    graph_engine::Shape s1 = graph_engine::Shape(3);
    graph_engine::Shape s2 = graph_engine::Shape(4);
    std::optional<graph_engine::Shape> result;

    s1[0] = 4; s1[1] = 5; s1[2] = 10;
    s2[0] = 2; s2[1] = 4; s2[2] = 1; s2[3] = 45;
    result = graph_engine::calculate_matmul_compatible_shape(s1, s2);

    ASSERT_TRUE(result.has_value());
    EXPECT_THAT(result.value(), ::testing::ElementsAre(2, 5, 10, 45));
}

