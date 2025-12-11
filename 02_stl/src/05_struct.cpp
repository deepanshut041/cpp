#include <iostream>
#include <vector>

using namespace std;

struct Ship {
    string Name;
    int NumAAGuns= 5;
    int NumCannons = 5;
    int NumMissiles = 5;
    int NumTorpedos = 5;
    int IsAnAircraftGun = true;
    int NumJets = 7;
    int NumHelis = 10;
};

int main(){
    Ship playerShip ;
    playerShip.Name = "Player Ship";

    Ship alliedShip { "Allied Ship", 6, 6, 7, 8, false, 0, 0 };

    Ship enemyShip = {"Enemy Ship", 4, 4, 4, 5, false, 0, 0};

    cout << "Name: " << playerShip.Name << " | Num of AAGuns: " << playerShip.NumAAGuns << " | Num of Jets: " << playerShip.NumJets << endl;
    cout << "Name: " << alliedShip.Name << " | Num of AAGuns: " << alliedShip.NumAAGuns << " | Num of Jets: " << alliedShip.NumJets << endl;
    cout << "Name: " << enemyShip.Name << " | Num of AAGuns: " << enemyShip.NumAAGuns << " | Num of Jets: " << enemyShip.NumJets << endl;
}