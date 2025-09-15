#include <gtest/gtest.h>
#include "15_longest_common_prefix.h"

class LongestCommonPrefixTest : public ::testing::Test {
protected:
    void assertPrefix(const std::vector<std::string>& input, const std::string& expected) {
        EXPECT_EQ(longestCommonPrefix(const_cast<std::vector<std::string>&>(input)), expected);
    }
};

TEST_F(LongestCommonPrefixTest, BasicTest) {
    assertPrefix({"flower", "flow", "flight"}, "fl");
}

TEST_F(LongestCommonPrefixTest, NoCommonPrefix) {
    assertPrefix({"dog", "racecar", "car"}, "");
}

TEST_F(LongestCommonPrefixTest, FullMatch) {
    assertPrefix({"test", "test", "test"}, "test");
}

TEST_F(LongestCommonPrefixTest, SingleString) {
    assertPrefix({"a"}, "a");
}


TEST_F(LongestCommonPrefixTest, MixedLengths) {
    assertPrefix({"interspecies", "interstellar", "interstate"}, "inters");
}