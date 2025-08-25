#include <iostream>

using namespace std;

class Sum{
    public:
      void printSum(const int &a, const int &b){
          int c = sum(a, b);
          cout << "Sum: " << c << endl;
      };
    private:
      int sum(const int &a, const int &b){
          return a + b;
      }
    protected:
};

class Rectangle{
    private:
        int width, height;
    public:
        Rectangle(int width, int height): width(width), height(height){};
        long area(){
            return this->width * this->height;
        }

        long perimeter();
};

long Rectangle::perimeter() {
    return 2 * (this->width + this->height);
}


int main(){
    Sum cl = Sum();
    cl.printSum(5, 6);

    Rectangle rect = Rectangle(5, 7);

    cout << "Area: " << rect.area() << endl;
    cout << "Perimeter: " << rect.perimeter() << endl;
}