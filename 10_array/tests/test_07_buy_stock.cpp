#include <gtest/gtest.h>
#include <vector>
#include "07_buy_stock.h"

class BuyStockTest : public ::testing::Test {
protected:
    void assertMaxProfit(const std::vector<int>& prices, int expected) {
        EXPECT_EQ(maxProfit(const_cast<std::vector<int>&>(prices)), expected);
    }
};

TEST_F(BuyStockTest, BasicTest) {
    assertMaxProfit({7, 1, 5, 3, 6, 4}, 5);
}

TEST_F(BuyStockTest, NoProfit) {
    assertMaxProfit({7, 6, 4, 3, 1}, 0);
}

TEST_F(BuyStockTest, SingleDay) {
    assertMaxProfit({5}, 0);
}

TEST_F(BuyStockTest, TwoDaysProfit) {
    assertMaxProfit({1, 5}, 4);
}

TEST_F(BuyStockTest, TwoDaysNoProfit) {
    assertMaxProfit({5, 1}, 0);
}