#include <iostream>

using namespace std;

enum class WEAPONS{
    MISSILE, // This has 0
    TORPEDO, // This has 1
    CANNON, // This has 2
    ANTI_AIRCRAFT, // This has 3
    RAILGUN, // This has 4
};

enum class WEAPONS_V2 {
    MISSILE, // This has 0
    TORPEDO, // This has 1
    CANNON, // This has 2
    ANTI_AIRCRAFT, // This has 3
    RAILGUN, // This has 4
};

int main(){
    WEAPONS torpedo = WEAPONS::MISSILE;
    WEAPONS_V2 torpedo_v2 = WEAPONS_V2::MISSILE;

    switch (torpedo) {
        case WEAPONS::MISSILE:
            cout << "Missile";
            break;
        case WEAPONS::TORPEDO:
            cout << "Torpedo";
            break;
        case WEAPONS::CANNON:
            cout << "cannon";
            break;
        case WEAPONS::ANTI_AIRCRAFT:
            cout << "anti aircraft";
            break;
        case WEAPONS::RAILGUN:
            cout << "Rail GUN";
            break;
    }
}