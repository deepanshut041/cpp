#include <string>
#include <vector>

using namespace std;

vector<string> createAndInsert()
{
    vector<string> coll; // create vector of strings
    coll.reserve(3); // reserve memory for 3 elements

    string s = "data"; // create string object
    
    coll.push_back(s); // insert string object
    coll.push_back(s+s); // insert temporary string
    coll.push_back(s); // insert string

    return coll; // return vector of strings
}

int main(){
    vector<string> v;
    v = createAndInsert();
}