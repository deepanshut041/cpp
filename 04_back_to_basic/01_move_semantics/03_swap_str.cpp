#include <iostream>
#include <string>
using namespace std;

void log(const char* label, const string& s) {
    cout << label << " @ " << static_cast<const void*>(&s)
         << " value='" << s << "'\n";
}

void mySwap(string& a, string& b) {
    cout << "\n--- inside mySwap ---\n";
    log("a (before)", a);
    log("b (before)", b);

    string tmp = std::move(a);   // move a → tmp
    log("tmp (after move from a)", tmp);
    log("a (after move)", a);

    a = std::move(b);            // move b → a
    log("a (after move from b)", a);
    log("b (after move)", b);

    b = std::move(tmp);          // move tmp → b
    log("b (after move from tmp)", b);
    log("tmp (after move)", tmp);

    cout << "--- end mySwap ---\n";
}

int main() {
    string a = "hello";
    string b = "world";

    cout << "Before swap:\n";
    log("a", a);
    log("b", b);

    mySwap(a, b);

    cout << "\nAfter swap:\n";
    log("a", a);
    log("b", b);
}
