#include <gtest/gtest.h>
#include <vector>
#include "06_rotate_array.h" // Include the header file where `rotate` is declared

class RotateArrayTest : public ::testing::Test {
protected:
    void assertRotation(const std::vector<int>& input, int k, const std::vector<int>& expected) {
        std::vector<int> nums = input; // Create a copy of the input to modify
        rotate(nums, k);
        EXPECT_EQ(nums, expected);
    }
};

TEST_F(RotateArrayTest, BasicTest) {
    assertRotation({1, 2, 3, 4, 5, 6, 7}, 3, {5, 6, 7, 1, 2, 3, 4});
}

TEST_F(RotateArrayTest, NegativeNumbers) {
    assertRotation({-1, -100, 3, 99}, 2, {3, 99, -1, -100});
}

TEST_F(RotateArrayTest, NoRotation) {
    assertRotation({1, 2, 3, 4}, 0, {1, 2, 3, 4});
}

TEST_F(RotateArrayTest, FullRotation) {
    assertRotation({1, 2, 3, 4}, 4, {1, 2, 3, 4});
}

TEST_F(RotateArrayTest, LargeK) {
    assertRotation({1, 2, 3, 4, 5}, 7, {4, 5, 1, 2, 3}); // k = 7 is equivalent to k = 2
}

TEST_F(RotateArrayTest, SingleElement) {
    assertRotation({1}, 3, {1});
}