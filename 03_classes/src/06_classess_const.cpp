#include <iostream>
#include <string>

class Item {
private:
    int value;
    std::string name;

public:
    Item(int v, std::string n) : value(v), name(std::move(n)) {}

    int&       Value()       { return value; }       // non-const version
    const int& Value() const { return value; }       // const version

    std::string&       Name()       { return name; }
    const std::string& Name() const { return name; }

    void setValue(int v) { value = v; }              // non-const
    int  getValue() const { return value; }          // const
};

// ---------------- functions ----------------

// takes object as const reference → can only call const methods
void PrintItem(const Item& it) {
    std::cout << "PrintItem: " << it.Value() << " " << it.Name() << "\n";

    // it.setValue(50);   // ❌ ERROR: cannot call non-const method
    // it.Name() = "X";   // ❌ ERROR: const overload prevents modification
}

// takes object as non-const reference → can call everything
void ModifyItem(Item& it) {
    it.setValue(99);         // OK
    it.Name() = "modified";  // OK
}

int main() {
    Item i(10, "hello");

    // pass to const function
    PrintItem(i);   // calls const overloads only

    // pass to non-const function
    ModifyItem(i);  // allows modifications
    PrintItem(i);   // see modifications

    // const object
    const Item ci(42, "fixed");
    PrintItem(ci);   // OK
    // ModifyItem(ci); // ❌ ERROR: cannot bind non-const reference to const object
}
