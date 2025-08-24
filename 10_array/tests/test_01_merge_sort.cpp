#include <gtest/gtest.h>
#include <vector>
#include "01_merge_sort.h"

class MergeSortedArrayTest : public ::testing::Test {
protected:
    void assertMerge(std::vector<int> nums1, int m, std::vector<int> nums2, int n, const std::vector<int>& expected) {
        merge(nums1, m, nums2, n);
        EXPECT_EQ(nums1, expected);
    }
};

TEST_F(MergeSortedArrayTest, BasicMerge) {
    assertMerge({1,2,3,0,0,0}, 3, {2,5,6}, 3, {1,2,2,3,5,6});
}

TEST_F(MergeSortedArrayTest, SecondArrayEmpty) {
    assertMerge({1}, 1, {}, 0, {1});
}

TEST_F(MergeSortedArrayTest, FirstArrayEmpty) {
    assertMerge({0}, 0, {1}, 1, {1});
}

TEST_F(MergeSortedArrayTest, BothArraysEmpty) {
    assertMerge({}, 0, {}, 0, {});
}

TEST_F(MergeSortedArrayTest, AllElementsFromSecondArraySmaller) {
    assertMerge({4,5,6,0,0,0}, 3, {1,2,3}, 3, {1,2,3,4,5,6});
}

TEST_F(MergeSortedArrayTest, AllElementsFromFirstArraySmaller) {
    assertMerge({1,2,3,0,0,0}, 3, {4,5,6}, 3, {1,2,3,4,5,6});
}

TEST_F(MergeSortedArrayTest, InterleavedMergeWithDuplicates) {
    assertMerge({1,3,5,0,0,0}, 3, {2,3,6}, 3, {1,2,3,3,5,6});
}

TEST_F(MergeSortedArrayTest, NegativeNumbers) {
    assertMerge({-3,-2,-1,0,0,0}, 3, {-5,-4,-3}, 3, {-5,-4,-3,-3,-2,-1});
}

TEST_F(MergeSortedArrayTest, MixedPositiveAndNegative) {
    assertMerge({-1,0,3,0,0,0}, 3, {-2,2,5}, 3, {-2,-1,0,2,3,5});
}
