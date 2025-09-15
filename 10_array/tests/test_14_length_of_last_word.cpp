#include <gtest/gtest.h>
#include "14_length_of_last_word.h"

class LengthOfLastWordTest : public ::testing::Test {
protected:
    void assertLength(const std::string& input, int expected) {
        EXPECT_EQ(lengthOfLastWord(input), expected);
    }
};

TEST_F(LengthOfLastWordTest, BasicTest) {
    assertLength("Hello World", 5);
}

TEST_F(LengthOfLastWordTest, LeadingAndTrailingSpaces) {
    assertLength("   fly me   to   the moon  ", 4);
}

TEST_F(LengthOfLastWordTest, MultipleWords) {
    assertLength("luffy is still joyboy", 6);
}

TEST_F(LengthOfLastWordTest, SingleWord) {
    assertLength("word", 4);
}

TEST_F(LengthOfLastWordTest, SpacesOnly) {
    assertLength("    ", 0); // Assuming no word means length 0
}