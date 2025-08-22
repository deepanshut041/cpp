#include <iostream>
#include <ostream>

using namespace std;

int main(){
    char a;

    cout << "Enter the char: " << ends;
    cin >> a;

    switch (a) {
        case 'a':
        case 'e':
        case 'i':
        case 'o':
        case 'u':
        case 'A':
        case 'E':
        case 'I':
        case 'O':
        case 'U':
            cout << a << " is Vowel" << endl;
            break;
        default:
            cout << a <<" is Constant" << endl;
            break;
    }

    return 0;

}