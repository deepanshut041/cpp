#include <gtest/gtest.h>
#include "14_factorial.h"


TEST(FactorialTest, Zero) {
    EXPECT_EQ(factorial(0), 1);
}

TEST(FactorialTest, SmallNumbers) {
    EXPECT_EQ(factorial(1), 1);
    EXPECT_EQ(factorial(2), 2);
    EXPECT_EQ(factorial(3), 6);
    EXPECT_EQ(factorial(4), 24);
    EXPECT_EQ(factorial(5), 120);
}

TEST(FactorialTest, LargerNumber) {
    EXPECT_EQ(factorial(10), 3628800);
}