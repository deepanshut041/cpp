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
        void setWidth(int width) {
            this->width = width;
        }
        void setHeight(int height) {
            this->height = height;
        }
        void setHeight();
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

void printInfo(Rectangle &rect, string name){
    cout << "Name: " << name << endl;
    cout << "Area: " << rect.area() << endl;
    cout << "Perimeter: " << rect.perimeter() << endl;
    cout << "Is Square: " << (rect.isSquare()? "Yes": "No") << endl << endl;
}

int main(){;
    Rectangle rect_1(5, 7);

    Rectangle *ptr_rect_1, *ptr_rect_2, *ptr_rect_3;

    // Make ptr_rect_1 point to the existing rectangle object rect_1 (on the stack)
    ptr_rect_1 = &rect_1;

    // Create a new Rectangle object on the heap (width=4, height=6)
    // and assign its address to ptr_rect_2
    ptr_rect_2 = new Rectangle(4, 6);

    // Allocate an array of 2 Rectangle objects on the heap
    // (first with width=2, height=5; second with width=3, height=6)
    // and assign the starting address to ptr_rect_3
    ptr_rect_3 = new Rectangle[2]{{2, 5}, {3, 6}};

    printInfo(rect_1, "rect_1");
    printInfo(*ptr_rect_1, "ptr_rect_1");
    printInfo(*ptr_rect_2, "ptr_rect_2");
    printInfo(ptr_rect_3[0], "ptr_rect_3 -> 0");
    printInfo(ptr_rect_3[1], "ptr_rect_3 -> 1");

    delete ptr_rect_2;
    delete ptr_rect_3;
}