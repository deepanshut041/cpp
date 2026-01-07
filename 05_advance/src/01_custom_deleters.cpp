#include <cstring>
#include <iostream>
#include <memory>

using namespace std;

struct Storage {
    shared_ptr<void> data;
    size_t bytes;

    Storage(size_t nbytes): bytes(nbytes) {
        void* raw = ::operator new(nbytes); //Allocate raw heap memory
        memset(raw, 0, nbytes); // Initialize memory to zero

        data = shared_ptr<void>(raw, [](void* p) {
            cout << "Deleting the memory of Storage 1!" << endl;
            ::operator delete(p);
        }); // Custom deleter for raw memory
    }
};

struct Storage2 {
    shared_ptr<void> data;
    size_t bytes;

    Storage2(size_t nbytes): bytes(nbytes) {
        void* raw = ::operator new(nbytes);
        memset(raw, 0, nbytes);

        data = shared_ptr<void>(raw, [](void* p) {
            cout << "Deleting the memory of Storage 2!" << endl;
            ::operator delete(p);
        });
    }

    [[nodiscard]] size_t size() const{
        return bytes;
    }

    [[nodiscard]] void* ptr() const {
        return data.get();
    }

    void resize(const size_t new_bytes) {
        const size_t old_bytes = bytes;

        void* raw = ::operator new(new_bytes);
        memset(raw, 0, new_bytes);

        size_t to_copy = min(old_bytes, new_bytes);
        if (data.get() && to_copy > 0) {
            std::memcpy(raw, data.get(), to_copy);
        }

        std::shared_ptr<void> new_data(raw, [](void* p) {
            cout << "Deleting the memory of Storage 2 Reset!" << endl;
            ::operator delete(p);
        });

        data = std::move(new_data);
        bytes = new_bytes;
    }
};

int main() {
    const Storage s(64);
    cout << "Allocated " << s.bytes << " bytes\n";

    Storage2 s2(72);

    cout << "\n[main] Current size: " << s2.size() << "\n";
    cout << "[main] Current ptr : " << s2.ptr() << "\n";

    s2.resize(60);

    cout << "\n[main] After resize size: " << s2.size() << "\n";
    cout << "[main] After resize ptr : " << s2.ptr() << "\n";

    cout << "\n---- Program End ----\n";
    return 0;
}