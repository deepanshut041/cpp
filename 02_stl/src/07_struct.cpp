#include <vector>
#include <iostream>
#include <string>

using namespace std;

struct Employee {
    int id;
    string name;
    int age;
    int wage;
};

struct Company {
    string name;
    Employee ceo;
    vector<Employee> managements;
};

// Print an Employee
void printEmployee(const Employee& e) {
    cout << "   [ID: " << static_cast<int>(e.id)   // cast int8_t so it doesn’t print weird chars
         << ", Name: " << e.name
         << ", Age: " << static_cast<int>(e.age)
         << ", Wage: $" << e.wage
         << "]\n";
}

// Print a Company
void printCompany(const Company& c) {
    cout << "Company: " << c.name << "\n";
    cout << "CEO:\n";
    printEmployee(c.ceo);

    cout << "Management:\n";
    for (const auto& m : c.managements) {
        printEmployee(m);
    }
}

int main() {
    Company c = {
        "ABC",
        { 1, "CEO", 34, 10000 },
        { { 2, "E001", 24, 1000 }, { 3, "E002", 26, 2000 } }
    };

    printCompany(c);
}