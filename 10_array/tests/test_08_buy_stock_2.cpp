#include <gtest/gtest.h>
#include <vector>
#include "08_buy_stock_2.h"

class MaxProfitTest : public ::testing::Test {
protected:
    void assertMaxProfit(const std::vector<int>& prices, int expected) {
        std::vector<int> input = prices; // Create a copy of the input
        int result = maxProfit(input);
        EXPECT_EQ(result, expected);
    }
};

class MaxProfitTest2 : public ::testing::Test {
  protected:
      void assertMaxProfit(const std::vector<int>& prices, int expected) {
          std::vector<int> input = prices; // Create a copy of the input
          int result = maxProfit2(input);
          EXPECT_EQ(result, expected);
      }
  };

TEST_F(MaxProfitTest, Example1) {
    assertMaxProfit({7, 1, 5, 3, 6, 4}, 7);
}

TEST_F(MaxProfitTest, Example2) {
    assertMaxProfit({1, 2, 3, 4, 5}, 4);
}

TEST_F(MaxProfitTest, Example3) {
    assertMaxProfit({7, 6, 4, 3, 1}, 0);
}

TEST_F(MaxProfitTest, SingleDay) {
    assertMaxProfit({5}, 0);
}

TEST_F(MaxProfitTest, NoProfit) {
    assertMaxProfit({5, 5, 5, 5}, 0);
}

TEST_F(MaxProfitTest, RandomTest) {
    assertMaxProfit({6, 1, 3, 2, 4, 7}, 7);
}

TEST_F(MaxProfitTest2, Example1) {
    assertMaxProfit({7, 1, 5, 3, 6, 4}, 7);
}

TEST_F(MaxProfitTest2, Example2) {
    assertMaxProfit({1, 2, 3, 4, 5}, 4);
}

TEST_F(MaxProfitTest2, Example3) {
    assertMaxProfit({7, 6, 4, 3, 1}, 0);
}

TEST_F(MaxProfitTest2, SingleDay) {
    assertMaxProfit({5}, 0);
}

TEST_F(MaxProfitTest2, NoProfit) {
    assertMaxProfit({5, 5, 5, 5}, 0);
}

TEST_F(MaxProfitTest2, RandomTest) {
    assertMaxProfit({6, 1, 3, 2, 4, 7}, 7);
}