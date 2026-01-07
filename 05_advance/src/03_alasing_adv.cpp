#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>

using namespace std;

struct Storage {
    shared_ptr<uint8_t> data;
    size_t bytes;
    size_t version;

    Storage(size_t nbytes) : bytes(nbytes), version(0) {
        void *raw = ::operator new(bytes);
        memset(raw, 0, nbytes);

        data = shared_ptr<uint8_t>(static_cast<uint8_t *>(raw), [](uint8_t *p) {
            cout << "Deleting the memory of Storage 1!" << endl;
            ::operator delete(p);
        });
    }
};

class SharedByteBuffer {
private:
    shared_ptr<Storage> storage_;
    shared_ptr<uint8_t> data_;
    size_t size_;
    size_t offset_;

public:
    SharedByteBuffer(size_t nbytes) : storage_(make_shared<Storage>(nbytes)),
                                      data_(storage_->data),
                                      size_(storage_->bytes),
                                      offset_(0) {
    };

    SharedByteBuffer(shared_ptr<Storage> storage, size_t offset, size_t len) : storage_(std::move(storage)),
                                                                               size_(len),
                                                                               offset_(offset) {
        uint8_t *view_ptr = storage_->data.get() + offset_;
        data_ = shared_ptr<uint8_t>(storage_->data, view_ptr);
    };

    [[nodiscard]] size_t bytes() const { return size_; }
    [[nodiscard]] size_t offset() const { return offset_; }

    [[nodiscard]] size_t storage_version() const { return storage_->version; }

    [[nodiscard]] SharedByteBuffer slice(size_t offset, size_t len) const {
        size_t new_offset = offset_ + offset;
        size_t max_len = storage_->bytes - new_offset;
        size_t final_len = min(len, max_len);
        return {storage_, new_offset, final_len};
    }

    void fill(uint8_t value) {
        memset(data_.get(), value, size_);
        storage_->version++;
    }

    void copy_from(const void* src, size_t n) {
        size_t to_copy = min(n, size_);
        memcpy(data_.get(), src, to_copy);
        storage_->version++;
    }

    string as_hex_string() const {
        ostringstream out;
        const uint8_t* p = data_.get();
        for (size_t i = 0; i < size_; i++) {
            out << hex << setw(2) << setfill('0')
                << (int)p[i];
            if (i + 1 != size_) out << " ";
        }
        return out.str();
    }
};

int main() {
    SharedByteBuffer buf(16);
    cout << buf.as_hex_string() << "\n";
    cout << "v=" << buf.storage_version() << "\n";

    auto slice = buf.slice(4, 6);
    slice.fill(0xAB);

    cout << "buf:   " << buf.as_hex_string() << "\n";
    cout << "slice: " << slice.as_hex_string() << "\n";

    cout << "v=" << buf.storage_version() << "\n";
}
