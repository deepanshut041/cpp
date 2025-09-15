#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int lengthOfLastWord(const string& s) {
    int mS = 0;
    int len = s.length();

    int r = 0;

    for(int i=0; i < len; i++){
        if (s[i] == ' '){
            mS = r > 0 ? r : mS;
            r = 0;
        } else {
            r++;
        }
    }

    mS = r > 0 ? r : mS;

    return mS;
}

#ifndef TESTING
int main() {
    string s1 = "Hello World";
    cout << "Length of Last Word (\"Hello World\"): " << lengthOfLastWord(s1) << endl;

    string s2 = "   fly me   to   the moon  ";
    cout << "Length of Last Word (\"   fly me   to   the moon  \"): " << lengthOfLastWord(s2) << endl;

    string s3 = "luffy is still joyboy";
    cout << "Length of Last Word (\"luffy is still joyboy\"): " << lengthOfLastWord(s3) << endl;

    string s4 = "word";
    cout << "Length of Last Word (\"word\"): " << lengthOfLastWord(s4) << endl;

    string s5 = "    ";
    cout << "Length of Last Word (\"    \"): " << lengthOfLastWord(s5) << endl;

    string s6 = "Today is a nice day";
    cout << "Length of Last Word (\"Today is a nice day\"): " << lengthOfLastWord(s6) << endl;

    return 0;
}
#endif