#include <gtest/gtest.h>
#include "graph/graph.h"

// Простейший тест на проверку математики
TEST(BasicAssertions, MathCheck) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}

TEST(ProjectLogic, SimpleTrue) {
    bool is_compiler_working = true;
    ASSERT_TRUE(is_compiler_working);
}

TEST(GraphEngine, ShapeInitialization) {
    graph_engine::Shape shape = graph_engine::Shape();
    ASSERT_EQ(shape.rank(), 0) << "ERR: rank after empty initialization is not 0";
    /*
    bool is_zero = true;
    for (int i = 0; i < graph_engine::MAX_VALUE_RANK; ++i) {
        is_zero = is_zero * shape[i];
    }
    EXPECT_EQ(is_zero, false) << "ERR: shape initialized with non-zero dims";
    */
}