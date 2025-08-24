#include <gtest/gtest.h>
#include <vector>
#include "03_remove_duplicates.h" // Replace with the actual header file name

class RemoveDuplicatesTest : public ::testing::Test {
protected:
    void assertRemoveDuplicates(std::vector<int> nums, const std::vector<int>& expected, int expectedK) {
        int k = removeDuplicates(nums);
        EXPECT_EQ(k, expectedK);
        std::vector<int> result(nums.begin(), nums.begin() + k);
        EXPECT_EQ(result, expected);
    }
};

TEST_F(RemoveDuplicatesTest, BasicTest) {
    assertRemoveDuplicates({1, 1, 2}, {1, 2}, 2);
}

TEST_F(RemoveDuplicatesTest, AllUnique) {
    assertRemoveDuplicates({0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, 5);
}

TEST_F(RemoveDuplicatesTest, AllDuplicates) {
    assertRemoveDuplicates({2, 2, 2, 2}, {2}, 1);
}

TEST_F(RemoveDuplicatesTest, MixedDuplicates) {
    assertRemoveDuplicates({0, 0, 1, 1, 1, 2, 2, 3, 3, 4}, {0, 1, 2, 3, 4}, 5);
}

TEST_F(RemoveDuplicatesTest, SingleElement) {
    assertRemoveDuplicates({1}, {1}, 1);
}

TEST_F(RemoveDuplicatesTest, EmptyArray) {
    assertRemoveDuplicates({}, {}, 0);
}

TEST_F(RemoveDuplicatesTest, LargeInput) {
    std::vector<int> nums(1000, 1);
    nums.insert(nums.end(), 1000, 2);
    nums.insert(nums.end(), 1000, 3);
    std::vector<int> expected = {1, 2, 3};
    assertRemoveDuplicates(nums, expected, 3);
}