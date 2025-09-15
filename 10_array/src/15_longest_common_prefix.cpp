#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

string longestCommonPrefix(vector<string>& strs) {
    auto prefix = strs.front();

    int r = prefix.length();

    for(int i = 1; i < strs.size(); i++){
        auto str = strs.at(i);
        int l = str.length();
        r = min(r, l);
        for (int j = 0; j < r; j++){
            if (str[j] != prefix[j]){
                r = j;
                break;
            }
        }
    }

    return prefix.substr(0, r);

}

#ifndef TESTING
int main() {
    vector<string> strs1 = {"flower", "flow", "flight"};
    cout << "Longest Common Prefix ([\"flower\", \"flow\", \"flight\"]): "
              << "\"" << longestCommonPrefix(strs1) << "\"" << endl;

    vector<string> strs2 = {"dog", "racecar", "car"};
    cout << "Longest Common Prefix ([\"dog\", \"racecar\", \"car\"]): "
              << "\"" << longestCommonPrefix(strs2) << "\"" << endl;

    vector<string> strs3 = {"interspecies", "interstellar", "interstate"};
    cout << "Longest Common Prefix ([\"interspecies\", \"interstellar\", \"interstate\"]): "
              << "\"" << longestCommonPrefix(strs3) << "\"" << endl;

    vector<string> strs4 = {"a"};
    cout << "Longest Common Prefix ([\"a\"]): "
              << "\"" << longestCommonPrefix(strs4) << "\"" << endl;

    vector<string> strs5 = {"ab", "a"};
    cout << "Longest Common Prefix ([\"ab\", \"a\"]): "
              << "\"" << longestCommonPrefix(strs5) << "\"" << endl;

    return 0;
}
#endif