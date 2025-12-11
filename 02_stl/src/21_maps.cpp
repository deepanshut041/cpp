#include <iostream>
#include <map>

using namespace std;

int main(){
    map<string, string> cars = {{"Mazada", "v4"}, {"Bmw", "v5"}, {"Raptor", "v6"}};

    cout << "Mazada is: " << cars["Mazada"] << "\n";

    cars["Mazada"] = "v8";
    cout << "Mazada is: " << cars.at("Mazada") << "\n";

    cars["Audi"] = "v2";
    cout << "Audi is: " << cars.at("Audi") << "\n";

    cout << "Has Audi: " << (cars.count("Audi")? "Yes" : "No") << endl;
    cars.erase("Audi");
    cout << "Has Audi: " << (cars.count("Audi")? "Yes" : "No") << endl;

    cout << "All Cars" << endl;
    for (const auto car : cars) {
        cout << car.first << " is: " << car.second << "\n";
    }
}