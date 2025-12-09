#include <gtest/gtest.h>
#include "hello/lib.hpp"


TEST(Addition, Basic) {
    EXPECT_EQ(hello::add(1, 2), 3);
    EXPECT_EQ(hello::add(-5, 5), 0);
}


TEST(Greet, DefaultAndName) {
    EXPECT_EQ(hello::greet(""), "Hello, world!");
    EXPECT_EQ(hello::greet("Deep"), "Hello, Deep!");
}