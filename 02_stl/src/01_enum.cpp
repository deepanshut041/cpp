#include <iostream>

using namespace std;

enum WEAPONS {
    WEAPONS_MISSILE, // This has 0
    WEAPONS_TORPEDO, // This has 1
    WEAPONS_CANNON, // This has 2
    WEAPONS_ANTI_AIRCRAFT, // This has 3
    WEAPONS_RAILGUN, // This has 4
};

enum WEAPONS_V2 {
    WEAPONS_V2_MISSILE = 10, // This has 10
    WEAPONS_V2_TORPEDO, // This has 11
    WEAPONS_V2_CANNON, // This has 12
    WEAPONS_V2_ANTI_AIRCRAFT, // This has 13
    WEAPONS_V2_RAILGUN, // This has 14
};

int main(){
    WEAPONS missile = WEAPONS_MISSILE;
    WEAPONS torpedo(WEAPONS_TORPEDO);
    WEAPONS cannon{WEAPONS_CANNON};

    cout << missile << ", " << torpedo << ", " << cannon << endl;

    WEAPONS_V2 missile_2 = WEAPONS_V2_MISSILE;
    WEAPONS_V2 torpedo_2(WEAPONS_V2_TORPEDO);
    WEAPONS_V2 cannon_2{WEAPONS_V2_CANNON};

    cout << missile_2 << ", " << torpedo_2 << ", " << cannon_2 << endl;

    // Static Casting may usefull while taking cin
    WEAPONS railgun = static_cast<WEAPONS>(4);
    cout << railgun << endl;
}