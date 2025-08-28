#include <gtest/gtest.h>
#include <vector>
#include "09_jump_game.h"

class CanJumpTest : public ::testing::Test {
protected:
    void assertCanJump(const std::vector<int>& nums, bool expected) {
        std::vector<int> input = nums; // Create a copy of the input
        bool result = canJump(input);
        EXPECT_EQ(result, expected);
    }
};

TEST_F(CanJumpTest, Example1) {
    assertCanJump({2, 3, 1, 1, 4}, true);
}

TEST_F(CanJumpTest, Example2) {
    assertCanJump({3, 2, 1, 0, 4}, false);
}

TEST_F(CanJumpTest, SingleElement) {
    assertCanJump({0}, true);
}

TEST_F(CanJumpTest, AllZeros) {
    assertCanJump({0, 0, 0, 0}, false);
}

TEST_F(CanJumpTest, LargeJump) {
    assertCanJump({10, 1, 1, 1, 1}, true);
}

TEST_F(CanJumpTest, EdgeCase) {
    assertCanJump({1, 0, 0, 0, 0}, false);
}