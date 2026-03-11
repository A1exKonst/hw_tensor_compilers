#include <gtest/gtest.h>

// Простейший тест на проверку математики
TEST(BasicAssertions, MathCheck) {
    EXPECT_STRNE("hello", "world");
    EXPECT_EQ(7 * 6, 42);
}

TEST(ProjectLogic, SimpleTrue) {
    bool is_compiler_working = true;
    ASSERT_TRUE(is_compiler_working);
}