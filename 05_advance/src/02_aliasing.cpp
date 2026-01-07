#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

class SliceView {
    shared_ptr<char> data_;
    size_t size_;
public:
    SliceView(size_t size) : size_(size) {
        cout << "Allocating Slice View Memory\n";
        data_ = shared_ptr<char>(new char[size], [] (const char* ptr) {
            cout << "Freeing Slice View Memory\n";
            delete[] ptr;
        });
    }

    SliceView(const shared_ptr<char> &data, size_t size, size_t offset) : size_(size){
        char* view_ptr = data.get() + offset;
        data_ = shared_ptr<char>(data, view_ptr);
    }

    [[nodiscard]]size_t size() const { return size_; }

    void print_as_string() const {
        cout << data_.get() << endl;
    }

    [[nodiscard]]char* get() const { return data_.get(); }

    [[nodiscard]] shared_ptr<char> owner_ptr() const { return data_; }
};

class ByteView {
    shared_ptr<byte> data_;
    size_t size_;
    size_t offset_;

public:
    // Owner constructor (allocates)
    ByteView(size_t size) : size_(size), offset_(0) {
        cout << "Allocating Byte View Memory\n";

        data_ = shared_ptr<byte>(new byte[size], [](byte* ptr) {
            cout << "Freeing Byte View Memory\n";
            delete[] ptr;
        });

        memset(data_.get(), 0, size_);
    }

    // Subrange constructor (aliasing, no copy)
    ByteView(shared_ptr<byte> owner_storage, size_t offset, size_t length)
        : size_(length), offset_(offset) {

        byte* view_ptr = owner_storage.get() + offset_;
        data_ = shared_ptr<byte>(owner_storage, view_ptr); // ✅ aliasing constructor
    }

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t offset() const { return offset_; }

    [[nodiscard]] byte* get() const { return data_.get(); }

    void fill(uint8_t value) {
        memset(data_.get(), value, size_);
    }

    void debug_dump() const {
        for (size_t i = 0; i < size_; i++) {
            cout << (int)to_integer<unsigned char>(data_.get()[i]) << " ";
        }
        cout << "\n";
    }

    [[nodiscard]] shared_ptr<byte> owner_ptr() const { return data_; }
};

int main() {

    auto owner = shared_ptr<char>(new char[100], [](const char* p) {
        cout << "Freeing owner\n";
        delete[] p;
    });

    strcpy(owner.get(), "Hello Aliasing!");

    char* view_ptr = owner.get() + 6; // Make a view starting at offset 6
    const shared_ptr<char> view(owner, view_ptr); // aliasing constructor

    cout << "Owner See: " << owner.get() << endl;
    cout << "Viewer See: " << view.get() << endl;

    SliceView slice{100};
    strcpy(slice.get(), "Hello SliceView");

    slice.print_as_string();

    SliceView view1(slice.owner_ptr(), 10, 6); // "SliceView"
    SliceView view2(slice.owner_ptr(), 5, 0);  // "Hello"

    cout << "View1: ";
    view1.print_as_string();

    cout << "View2: ";
    view2.print_as_string();

    ByteView b_owner(16);
    b_owner.fill(1);

    cout << "Owner: ";
    b_owner.debug_dump();

    ByteView b_view(b_owner.owner_ptr(), 4, 6); // view [4..9]
    b_view.fill(9);

    cout << "Owner after sub fill: ";
    b_owner.debug_dump();

    cout << "Sub: ";
    b_view.debug_dump();

    return 0;
}