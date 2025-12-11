#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// Typedef vs using demo
typedef float velocity_t;          // typedef style
using speed_t = float;             // using alias style

// Example typedef and using for same type
typedef vector<pair<string, int>> PairList_t;
using PairList2_t = vector<pair<string, int>>;


int main() {
    // Aliases in action
    velocity_t vehicleVelocity = 250.0f;
    speed_t    vehicleSpeed    = 250.0f;

    if (vehicleVelocity == vehicleSpeed) {
        cout << "Velocity and Speed are comparable (both are float aliases)" << endl<< endl;
    }

    // Using typedef list
    PairList_t table{{"Alpha", 1}, {"Beta", 2}, {"Gamma", 3}};

    // Using using-alias list
    PairList2_t table2{{"Delta", 4}, {"Epsilon", 5}};

    // Merge them
    table.insert(table.end(), table2.begin(), table2.end());

    // Print the combined table
    cout << "Data Table:\n";
    for (const auto& [name, val] : table) {
        cout << " - " << setw(8) << left << name << " : " << val << endl;
    }

    return 0;
}