#include <iostream>

using namespace std;

int main(){
    int a, b, c;

    cout << "Enter A: " << ends;
    cin >> a;

    cout << "Enter B: " << ends;
    cin >> b;

    cout << "Enter C: " << ends;
    cin >> c;

    if (a > b) {
        if (a > c) {
            cout << a << " is the greatest number" << endl;
        } else {
            cout << c << " is the greatest number" << endl;
        }
    } else {
        if (b > c) {
            cout << b << " is the greatest number" << endl;
        } else {
            cout << c << " is the greatest number" << endl;
        }
    }
}