#include <cstdlib>
#include <iostream>
#include <string>
#include <memory>
#include <vector>

using namespace std;

// Return By Value
int squareValue(int x){
    int value = x * x;

    return value;
}

struct Weapon{
    string name;
    int maxAmmo;
    float rateOfFire;
};

// Return by refrence
vector<Weapon>* createWeapons(int n){
    auto weapons = new vector<Weapon>(n);

    for (auto &weapon: *weapons){
        weapon.name = "Name " + to_string(rand() % 1000);
        weapon.maxAmmo = rand() % 100;
        weapon.rateOfFire = rand() % 100;
    }

    return weapons;
}

unique_ptr<vector<Weapon>> createWeapons2(int n){
    auto weapons = make_unique<vector<Weapon>>(n);

    for (auto &weapon: *weapons){
        weapon.name = "Name " + to_string(rand() % 1000);
        weapon.maxAmmo = rand() % 100;
        weapon.rateOfFire = rand() % 100;
    }

    return weapons;
}


// Pass by refrence
void printWeapons(const vector<Weapon> &weapons) {
    cout << endl << "Printing Weapons"<< endl << endl;

    for (const auto &weapon : weapons) {
        cout << "Name: " << weapon.name;
        cout << " | Max Ammo: " << weapon.maxAmmo;
        cout << " | Rate of Fire: " << weapon.rateOfFire << endl;
    }
}

int main(){
    int x = 5;

    // Return By Value
    cout << x << " Square is: " << squareValue(x) << endl;

    auto weapons = createWeapons(x * 4);
    printWeapons(*weapons);

    auto weapons2 = createWeapons2(x * 4);
    printWeapons(*weapons2);

    delete weapons;
    weapons = nullptr;

    // weapon2 doesn't need deleting it automatically cleanups even if something throws up. 
}