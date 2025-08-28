#include <iostream>

using namespace std;

static void logAddr(const char* tag, const string& s) {
  cout << tag << " @ " << static_cast<const void*>(&s)
       << "  value='" << s << "'\n";
}

static void foo(const string& s) { cout << "foo(const&): "; logAddr("arg", s); }
static void foo(string& s)       { cout << "foo(&):      "; logAddr("arg", s); }
static void foo(string&& s)      { cout << "foo(&&):     "; logAddr("arg", s); }

template <typename T>
static void callFoo(T&& x) {
  // Forward as rvalue if caller passed rvalue, lvalue if lvalue:
  foo(std::forward<T>(x));
}

int main() {
  string v = "v";
  const string c = "c";

  cout << "v     @ " << static_cast<const void*>(&v) << "  value='" << v << "'\n";
  cout << "c     @ " << static_cast<const void*>(&c) << "  value='" << c << "'\n";

  callFoo(v);                 // T = string&, forwards as lvalue  -> foo(&)
  callFoo(c);                 // T = const string&, forwards      -> foo(const&)
  callFoo(string{"tmp"});     // T = string, forwards as rvalue   -> foo(&&)
  callFoo(std::move(v));      // T = string, forwards as rvalue   -> foo(&&)

  cout << "v     @ " << static_cast<const void*>(&v) << "  value='" << v << "'\n";
  cout << "c     @ " << static_cast<const void*>(&c) << "  value='" << c << "'\n";
}
