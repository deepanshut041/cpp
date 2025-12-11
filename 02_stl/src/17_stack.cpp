#include <algorithm>
#include <iostream>
#include <stack>

using namespace std;

int main(){

    stack<string> cars;

    cars.push("Volvo");
    cars.push("BMW");
    cars.push("Ford");
    cars.push("Mazda");

    cout << "Total Cars -> " << cars.size() << endl;

    while (!cars.empty()){
        cout << cars.top() << endl;
        cars.pop();
    }

    cout << "Total Cars -> " << cars.size() << endl;
    return 0;

}