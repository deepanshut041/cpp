#include <cmath>
#include <iostream>

using namespace std;

struct Rectangle {
    double length = 1.0;
    double width = 1.0;
};

struct Vector3D{
    float x;
    float y;
    float z;
};

void PrintRectagleInfo(Rectangle r){
    cout << "Print Rectangle Info" << endl;
    cout << "Length: " << r.length << endl;
    cout << "Width: " << r.width << endl;
    cout << "Area: " << (r.length * r.width) << endl;
    cout << "Circumference: " << (2 * (r.length + r.width)) << endl;
}

Vector3D CreateUnitVector() {
    Vector3D unitVector = {1.0, 1.0, 1.0};
    return unitVector;
}

int main(){
    Rectangle r1 = { 4.0, 5.0};
    Rectangle r2 = { 7.0, 8.0};

    PrintRectagleInfo(r1);
    cout << endl;
    PrintRectagleInfo(r2);
    cout << endl << endl;

    Vector3D unitVector = CreateUnitVector();
    if(unitVector.x == 1.0 && unitVector.y == 1.0 && unitVector.z == 1.0){
        cout << "Variable is an unit vector" << endl;
    }
}