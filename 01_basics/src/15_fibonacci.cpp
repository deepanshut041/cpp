#include <iostream>
#include <vector>

using namespace std;

vector<int> fibonacci(int n) {
    vector <int> fib;
    if (n <= 0) return fib;

    fib.push_back(0);
    if(n == 1) return fib;

    fib.push_back(1);
    for (int i=2; i < n; i++){
        fib.push_back(fib[i - 1] + fib[i - 2]);
    }

    return fib;
}

#ifndef TESTING
int main() {
    int n;
    cout << "Enter n: " << ends;
    cin >> n;

    vector<int> fib = fibonacci(n);

    cout << "Fibonacci sequence: [" << ends;

    for(int f: fib){
        cout << f << ", " <<ends;
    }

    cout << "]" << endl;

    return 0;
}
#endif