#include <iostream>
#include <string>
#include <vector>

using namespace std;

int romanMapping(char c) {
    switch (c) {
        case 'I': return 1;
        case 'V': return 5;
        case 'X': return 10;
        case 'L': return 50;
        case 'C': return 100;
        case 'D': return 500;
        case 'M': return 1000;
    }
    return 0;
}

int romanToInt(string s) {
    int n = s.length();
    int v = 0;

    for (int i = 0; i < n; i++) {
        int curr = romanMapping(s[i]);
        int next = (i + 1 < n) ? romanMapping(s[i + 1]) : 0;

        if (curr < next) {
            v -= curr;
        } else {
            v += curr;
        }
    }

    return v;
}

int main(){
    cout << "III: " << romanToInt("III") << endl;
    cout << "LVIII: " << romanToInt("LVIII") << endl;
    cout << "MCMXCIV: " << romanToInt("MCMXCIV") << endl;
}