#include <iostream>

using namespace std;

class Rectangle{
    private:
        int width, height;
    public:
        Rectangle(int width, int height): width(width), height(height){};
        Rectangle(int width): width(width), height(width){};
        Rectangle() {};
        long area();
        long perimeter();
        bool isSquare();
        int getWidth() { return this -> width;};
        int getHeight() { return this -> height;};
        Rectangle operator+(const Rectangle &param);
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

Rectangle Rectangle::operator+(const Rectangle &param) {
    return Rectangle(width + param.width, height + param.height);
}

Rectangle operator-(Rectangle &lhs, Rectangle &rhs) {
    return Rectangle(lhs.getWidth() - rhs.getWidth(), lhs.getHeight() - rhs.getHeight());
}


void printInfo(Rectangle &rect, string name){
    cout << "Name: " << name << endl;
    cout << "Width: " << rect.getWidth() << endl;
    cout << "Height: " << rect.getHeight() << endl;
    cout << "Area: " << rect.area() << endl;
    cout << "Perimeter: " << rect.perimeter() << endl;
    cout << "Is Square: " << (rect.isSquare()? "Yes": "No") << endl << endl;
}

int main(){;
    Rectangle rect_a = Rectangle(5, 7);
    printInfo(rect_a, "rect_a");

    Rectangle rect_b = Rectangle(2, 3);
    printInfo(rect_b, "rect_b");

    auto react_sum = rect_a + rect_b;
    printInfo(react_sum, "react_sum");

    auto react_diff = rect_a - rect_b;
    printInfo(react_diff, "react_diff");


}