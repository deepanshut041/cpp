#include <iostream>
#include <set>

using namespace std;

int main() {
    set<string> cars = {"Volvo", "BMW", "Ford", "Mazda"};

    cout << "Total Cars -> " << cars.size() << endl;
    for (const auto car : cars) {
        cout << car << endl;
    }

    cars.insert("Tesla");

    cout << "Total Cars -> " << cars.size() << endl;
    for (const auto car : cars) {
        cout << car << endl;
    }

    cars.erase("Volvo");

    cout << endl
         << "Total Cars -> " << cars.size() << endl;
    for (const auto car : cars) {
        cout << car << endl;
    }

    // Keep sorted variabel
    cout << endl << "Sorted Cars " << endl;
    set<string, greater<string>> cars2 = {"Volvo", "BMW", "Ford", "Mazda"};
    for (const auto car : cars2) {
        cout << car << "\n";
    }

    // Check element is present or not
    if (cars2.count("Tesla")) {
        cout << "Tesla is present Cars 2" << endl;
    } else {
        cout << "Tesla is not present Cars 2" << endl;
    }
}