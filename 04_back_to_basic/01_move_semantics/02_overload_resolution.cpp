#include <iostream>
#include <string>
using namespace std;

static void logAddr(const char* tag, const string& s) {
  cout << tag << " @ " << static_cast<const void*>(&s)
       << "  value='" << s << "'\n";
}

static void foo(const string& s) { cout << "foo(const&): "; logAddr("arg", s); }
static void foo(string& s)       { cout << "foo(&):      "; logAddr("arg", s); }
static void foo(string&& s)      { cout << "foo(&&):     "; logAddr("arg", s); }

int main() {
  string v = "v";
  const string c = "c";

  cout << "v     @ " << static_cast<const void*>(&v) << "  value='" << v << "'\n";
  cout << "c     @ " << static_cast<const void*>(&c) << "  value='" << c << "'\n";

  foo(v);                       // binds to &: same address as v
  foo(c);                       // binds to const&: same address as c
  foo(string{"tmp"});           // binds to &&: address of a temporary
  foo(std::move(v));            // binds to &&: same address as v

  cout << "v     @ " << static_cast<const void*>(&v) << "  value='" << v << "'\n";
  cout << "c     @ " << static_cast<const void*>(&c) << "  value='" << c << "'\n";
}
