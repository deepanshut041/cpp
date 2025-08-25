#include <iostream>

using namespace std;

class MyClass{
    public:
      static int variable;
      MyClass(){
          variable++;
      };
};

int MyClass::variable = 0;

int main(){
    MyClass a;
    cout << MyClass::variable << "\n";

    MyClass b[5];
    cout << MyClass::variable << "\n";

    MyClass* c = new MyClass();
    cout << MyClass::variable << "\n";

    delete c;

}