#include <gtest/gtest.h>
#include <vector>
#include "10_jump_game2.h"

class MinJumpsTest : public ::testing::Test {
protected:
    void assertMinJumps(const std::vector<int>& nums, int expected) {
        std::vector<int> input = nums; // Create a copy of the input
        int result = minJumps(input);
        EXPECT_EQ(result, expected);
    }
};

TEST_F(MinJumpsTest, Example1) {
    assertMinJumps({2, 3, 1, 1, 4}, 2);
}

TEST_F(MinJumpsTest, Example2) {
    assertMinJumps({2, 3, 0, 1, 4}, 2);
}

TEST_F(MinJumpsTest, SingleElement) {
    assertMinJumps({0}, 0);
}

TEST_F(MinJumpsTest, LargeJump) {
    assertMinJumps({10, 1, 1, 1, 1}, 1);
}

TEST_F(MinJumpsTest, MultipleJumps) {
    assertMinJumps({1, 1, 1, 1, 1}, 4);
}

TEST_F(MinJumpsTest, EdgeCase) {
    assertMinJumps({1, 2, 3, 4, 5}, 3);
}