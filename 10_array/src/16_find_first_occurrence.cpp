#include <iostream>
#include <string>

using namespace std;

int strStr(const string& haystack, const string& needle){
    if (needle.empty()) return 0;

    int nl = needle.length();
    int hl = haystack.length();

    for (int i = 0; i <= hl - nl; ++i) {
        int r = 0;

        for (int j = 0; j < nl; ++j) {
            if (haystack[i + j] != needle[j]) {
                break;
            }
            ++r;
        }

        if (r == nl) {
            return i;
        }
    }

    return -1;
}


#ifndef TESTING
int main() {
    string haystack1 = "sadbutsad";
    string needle1 = "sad";
    cout << "Index of First Occurrence (\"sadbutsad\", \"sad\"): "
              << strStr(haystack1, needle1) << endl;

    string haystack2 = "leetcode";
    string needle2 = "leeto";
    cout << "Index of First Occurrence (\"leetcode\", \"leeto\"): "
              << strStr(haystack2, needle2) << endl;

    string haystack3 = "hello";
    string needle3 = "ll";
    cout << "Index of First Occurrence (\"hello\", \"ll\"): "
              << strStr(haystack3, needle3) << endl;

    string haystack4 = "aaaaa";
    string needle4 = "bba";
    cout << "Index of First Occurrence (\"aaaaa\", \"bba\"): "
              << strStr(haystack4, needle4) << endl;

    string haystack5 = "a";
    string needle5 = "a";
    cout << "Index of First Occurrence (\"a\", \"a\"): "
              << strStr(haystack5, needle5) << endl;

    return 0;
}
#endif