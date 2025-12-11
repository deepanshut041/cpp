#include <iostream>

using namespace std;


struct Weapon {
    int maxAmmo;
    float  rateOffFire;
};


void printWeapon(Weapon* weapon){
    if (!weapon) { cerr << "null Weapon*\n"; return; }
    cout <<"Max Ammo: " <<(*weapon).maxAmmo << " | Rate of fire: " << (*weapon).rateOffFire << endl;
}

void fireWeapon(Weapon* weapon){
    if (!weapon) return;
    (*weapon).maxAmmo--;
}

void fireWeapon2(Weapon* weapon){
    if (!weapon) return;
    weapon -> maxAmmo--;
}

int main(){
    Weapon weapon = {11, 0.2};
    printWeapon(&weapon);
    fireWeapon(&weapon);
    printWeapon(&weapon);
    fireWeapon2(&weapon);
    printWeapon(&weapon);
}