#include <iostream>
#include <vector>

using namespace std;

typedef float velocity_t;
using speed_t = float;

// rint will automaticly assingned a leng absed on system type
#ifdef INT_LEAST8_MIN
typedef char rint8_t;
typedef int rint16_t;
typedef long rint32_t;
#else
typedef char rint8_t;
typedef int rint16_t;
typedef long rint32_t;
#endif

typedef vector<pair<string, int>> PairList_t;
using PairList2_t = vector<pair<string, int>> ;

int main(){

    velocity_t vechileVelocity = 250.0f;
    speed_t vechileSpeed = 250.0f;

    if (vechileVelocity == vechileSpeed){
        cout << "Yes u can compare both becuase both are float underneath";
    }

    PairList_t simpleTable;
}