#include <iostream>

using namespace std;

class Rectangle{
    private:
        int width, height;
    public:
        Rectangle(int width, int height): width(width), height(height){};
        Rectangle(int width): width(width), height(width){};
        long area();
        long perimeter();
        bool isSquare();
};

long Rectangle::perimeter() {
    return 2 * (this->width + this->height);
}

long Rectangle::area() {
    return this->width * this->height;
}

bool Rectangle::isSquare(){
    return this->width == this->height;
}


int main(){;

    Rectangle rect = Rectangle(5, 7);

    cout << "Area: " << rect.area() << endl;
    cout << "Perimeter: " << rect.perimeter() << endl;
    cout << "Is Square: " << (rect.isSquare()? "Yes": "No") << endl;

    Rectangle square = Rectangle(5);

    cout << "Area: " << square.area() << endl;
    cout << "Perimeter: " << square.perimeter() << endl;
    cout << "Is Square: " << (square.isSquare()? "Yes": "No") << endl;
}