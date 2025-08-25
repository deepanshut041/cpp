#include <iostream>
#include <deque>

using namespace std;


int main(){
    deque<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};


    cout << "Total Cars -> " << cars.size() << endl;
    for (const auto car: cars){
        cout << car << endl;
    }

    cout << "First Car: " << cars.front() << endl;
    cout << "3rd Car: " << cars.at(2) << endl;
    cout << "Last Car: " << cars.back() << endl;

    cars.push_front("Tesla");
    cars.push_back("VW");

    cout << "Total Cars -> " << cars.size() << endl;
    for (const auto car: cars){
        cout << car << endl;
    }

    cars.pop_front();
    cars.pop_back();
    cout << "Total Cars -> " << cars.size() << endl;
    for (const auto car: cars){
        cout << car << endl;
    }

    return 0;
}