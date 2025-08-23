#include <gtest/gtest.h>
#include <vector>
#include <algorithm> // For std::sort
#include "02_remove_element.h"

class RemoveElementTest : public ::testing::Test {
protected:
    void assertRemove(std::vector<int> nums, int val, const std::vector<int>& expected, int expectedK) {
        int k = removeElement(nums, val);
        EXPECT_EQ(k, expectedK);
        std::sort(nums.begin(), nums.begin() + k); // Sort the first k elements for comparison
        std::vector<int> result(nums.begin(), nums.begin() + k);
        EXPECT_EQ(result, expected);
    }
};

TEST_F(RemoveElementTest, BasicRemove) {
    assertRemove({3, 2, 2, 3}, 3, {2, 2}, 2);
}

TEST_F(RemoveElementTest, RemoveAllOccurrences) {
    assertRemove({2, 2, 2, 2}, 2, {}, 0);
}

TEST_F(RemoveElementTest, NoOccurrences) {
    assertRemove({1, 2, 3, 4, 5}, 6, {1, 2, 3, 4, 5}, 5);
}

TEST_F(RemoveElementTest, EmptyArray) {
    assertRemove({}, 1, {}, 0);
}

TEST_F(RemoveElementTest, MixedOccurrences) {
    assertRemove({0, 1, 2, 2, 3, 0, 4, 2}, 2, {0, 0, 1, 3, 4}, 5);
}

TEST_F(RemoveElementTest, SingleElementToRemove) {
    assertRemove({1}, 1, {}, 0);
}

TEST_F(RemoveElementTest, SingleElementToKeep) {
    assertRemove({1}, 2, {1}, 1);
}

TEST_F(RemoveElementTest, NewTestCase) {
    assertRemove({4, 5}, 4, {5}, 1);
}