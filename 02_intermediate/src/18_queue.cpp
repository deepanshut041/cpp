#include <iostream>
#include <queue>

using namespace std;


int main(){
    queue<string> cars;

    cars.push("Volvo");
    cars.push("BMW");
    cars.push("Ford");
    cars.push("Mazda");

    cout << "Total Cars -> " << cars.size() << endl;

    while (!cars.empty()){
        cout << cars.front() << endl;
        cars.pop();
    }

    cout << "Total Cars -> " << cars.size() << endl;
    return 0;
}