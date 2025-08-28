#include <iostream>
#include <vector>
#include <string>
using namespace std;

static string getData() {
  return "data";
}

static void print_state(const vector<string>& coll, const string& s, const char* label) {
  cout << "\n--- " << label << " ---\n";
  cout << "s: '" << s << "' (size=" << s.size() << ")\n";
  cout << "coll(size=" << coll.size() << "): [";
  for (size_t i = 0; i < coll.size(); ++i) {
    if (i) cout << ", ";
    cout << "'" << coll[i] << "'";
  }
  cout << "]\n";
}

int main() {
  vector<string> coll;
  coll.reserve(3);

  string s{getData()};        // s is an lvalue
  print_state(coll, s, "after constructing s");

  coll.push_back(s);          // copies (lvalue)
  print_state(coll, s, "after push_back(s)");

  coll.push_back(getData());  // moves (temporary is rvalue)
  print_state(coll, s, "after push_back(getData())");

  coll.push_back(std::move(s)); // moves (we say: I don't need s here)
  print_state(coll, s, "after push_back(std::move(s))");
}
