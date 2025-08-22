#include "13_sum_of_n_natural.h"
#include <gtest/gtest.h>

TEST(SumTest, Zero) {
    EXPECT_EQ(sumOfFirstNNaturalNumbers(0), 0);
}

TEST(SumTest, SmallNumbers) {
    EXPECT_EQ(sumOfFirstNNaturalNumbers(1), 1);
    EXPECT_EQ(sumOfFirstNNaturalNumbers(5), 15);
    EXPECT_EQ(sumOfFirstNNaturalNumbers(10), 55);
}

TEST(SumTest, LargerNumber) {
    EXPECT_EQ(sumOfFirstNNaturalNumbers(100), 5050);
}