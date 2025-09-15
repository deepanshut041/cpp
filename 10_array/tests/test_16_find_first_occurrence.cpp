#include <gtest/gtest.h>
#include "16_find_first_occurrence.h"

class FindFirstOccurrenceTest : public ::testing::Test {
protected:
    void assertIndex(const std::string& haystack, const std::string& needle, int expected) {
        EXPECT_EQ(strStr(haystack, needle), expected);
    }
};

TEST_F(FindFirstOccurrenceTest, BasicTest) {
    assertIndex("sadbutsad", "sad", 0);
}

TEST_F(FindFirstOccurrenceTest, NoOccurrence) {
    assertIndex("leetcode", "leeto", -1);
}

TEST_F(FindFirstOccurrenceTest, MiddleOccurrence) {
    assertIndex("hello", "ll", 2);
}

TEST_F(FindFirstOccurrenceTest, NoMatch) {
    assertIndex("aaaaa", "bba", -1);
}

TEST_F(FindFirstOccurrenceTest, SingleCharacterMatch) {
    assertIndex("a", "a", 0);
}

TEST_F(FindFirstOccurrenceTest, EmptyNeedle) {
    assertIndex("abc", "", 0); // Assuming empty needle always matches at index 0
}