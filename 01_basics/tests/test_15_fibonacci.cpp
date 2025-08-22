#include <gtest/gtest.h>
#include <vector>
#include "15_fibonacci.h"

using std::vector;

TEST(FibonacciTest, NonPositiveN) {
    EXPECT_TRUE(fibonacci(0).empty());
    EXPECT_TRUE(fibonacci(-5).empty());
}

TEST(FibonacciTest, NEqualsOne) {
    EXPECT_EQ(fibonacci(1), (vector<int>{0}));
}

TEST(FibonacciTest, NEqualsTwo) {
    EXPECT_EQ(fibonacci(2), (vector<int>{0, 1}));
}

TEST(FibonacciTest, SmallSequences) {
    EXPECT_EQ(fibonacci(5), (vector<int>{0, 1, 1, 2, 3}));
    EXPECT_EQ(fibonacci(10), (vector<int>{0, 1, 1, 2, 3, 5, 8, 13, 21, 34}));
}

TEST(FibonacciTest, SizeMatchesN) {
    for (int n : {0, 1, 2, 3, 10}) {
        EXPECT_EQ(fibonacci(n).size(), n > 0 ? static_cast<size_t>(n) : 0u);
    }
}