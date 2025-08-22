#include <iostream>

using namespace std;

int sumOfFirstNNaturalNumbers(int n) {
    return (n * (n + 1)) / 2;
}

#ifndef TESTING
int main(){
    int a;

    cout << "Enter the number: " << ends;
    cin >> a;

    cout << "Sum of first an natural number is " << sumOfFirstNNaturalNumbers(a) << endl;

    return 0;
}
#endif