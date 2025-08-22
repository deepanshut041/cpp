#include <iostream>

using namespace std;

int factorial(int n) {
    int f = 1;
    for (int i = 2; i <= n; i++){
        f = f * i;
    }

    return f;
}

#ifndef TESTING
int main(){
    int n;

    cout << "Enter Number: " << ends;
    cin >> n;

    int f = factorial(n);

    cout << "Factorial: " << f << endl;

    return 0;
}
#endif