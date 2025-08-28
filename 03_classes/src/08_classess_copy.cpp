#include <iostream>
#include <string>

using namespace std;

class MyClass {
    string *ptrString;
    public:
        MyClass(const string& str): ptrString(new string(str)){};
        MyClass(const MyClass& param): ptrString(new string((*param.ptrString + " Is a Copy"))){};
        ~MyClass() { delete ptrString; }
        string getStr() { return *ptrString; }
};

int main(){
    MyClass class_1("Sample");
    MyClass class_2 = class_1;

    cout << "Class 1's content: " << class_1.getStr() << endl;
    cout << "Class 2's content: " << class_2.getStr() << endl;

}