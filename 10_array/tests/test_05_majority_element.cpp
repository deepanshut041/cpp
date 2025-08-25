#include <gtest/gtest.h>
#include <vector>
#include "05_majority_element.h"

class MajorityElementTest : public ::testing::Test {
  protected:
      void assertMajorityElement(const std::vector<int>& nums, int expected) {
          int result = majorityElement(nums);
          EXPECT_EQ(result, expected);
      }
  };

TEST_F(MajorityElementTest, BasicTest) {
    assertMajorityElement({3, 2, 3}, 3);
}

TEST_F(MajorityElementTest, MajorityAtEnd) {
    assertMajorityElement({2, 2, 1, 1, 1, 2, 2}, 2);
}

TEST_F(MajorityElementTest, SingleElement) {
    assertMajorityElement({1}, 1);
}

TEST_F(MajorityElementTest, AllSameElement) {
    assertMajorityElement({5, 5, 5, 5}, 5);
}

TEST_F(MajorityElementTest, LargeInput) {
    std::vector<int> nums(1000, 1);
    nums.insert(nums.end(), 500, 2);
    assertMajorityElement(nums, 1);
}

TEST_F(MajorityElementTest, NegativeNumbers) {
    assertMajorityElement({-1, -1, -1, 2, 2}, -1);
}

TEST_F(MajorityElementTest, MixedNumbers) {
    assertMajorityElement({1, 1, 2, 2, 2, 1, 1, 1}, 1);
}