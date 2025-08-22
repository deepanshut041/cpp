#include <iostream>

using namespace std;

int main(){
    int y;

    cout << "Year: " << ends;
    cin >> y;

    cout << y << " is a" << ((y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)) ? " Leap Year" : " Non Leap year") << endl;
}