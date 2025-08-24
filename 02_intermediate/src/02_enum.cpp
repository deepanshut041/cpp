#include <iostream>

using namespace std;

enum FILE_STATUS {
    FILE_STATUS_SUCCESS = 1,
    FILE_STATUS_ERROR_NOT_FOUND = -1,
    FILE_STATUS_ERROR_READ_FILE = -2,
    FILE_STATUS_ERROR_PARSE_FILE = -3,
    FILE_STATUS_ERROR_CLOSE_FILE = -4,
};

class FileException : public exception {
  private:
    string message;
    FILE_STATUS status;

  public:
    FileException(string msg, FILE_STATUS s) : message(msg), status(s) {}
    const char *what() {
        return message.c_str();
    }
    FILE_STATUS code() {
        return status;
    }
};

class FileNotFoundException : public FileException {
  public:
    FileNotFoundException(const std::string &details = "")
        : FileException("File not found" + (details.empty() ? "" : (": " + details)),
                        FILE_STATUS_ERROR_NOT_FOUND) {}
};

class FileReadException : public FileException {
  public:
    FileReadException(const std::string &details = "")
        : FileException("Failed to read file" + (details.empty() ? "" : (": " + details)),
                        FILE_STATUS_ERROR_READ_FILE) {}
};

class FileParseException : public FileException {
  public:
    FileParseException(const std::string &details = "")
        : FileException("Failed to parse file" + (details.empty() ? "" : (": " + details)),
                        FILE_STATUS_ERROR_PARSE_FILE) {}
};

class FileCloseException : public FileException {
  public:
    FileCloseException(const std::string &details = "")
        : FileException("Failed to close file" + (details.empty() ? "" : (": " + details)),
                        FILE_STATUS_ERROR_CLOSE_FILE) {}
};

void openFile(string& filename){
    if(rand() % 10 < 6){
        cout << "File Opened successfully" << endl;
    } else{
        throw FileNotFoundException(filename);
    }
};

void readFile(string& filename){
    if(rand() % 10 < 6){
        cout << "Read File successfully" << endl;
    } else{
        throw FileReadException(filename);
    }
};

void parseFile(string& filename){
    if(rand() % 10 < 6){
        cout << "Parse File successfully" << endl;
    } else{
        throw FileParseException(filename);
    }
};

void closeFile(string& filename){
    if(rand() % 10 < 6){
        cout << "Closed File successfully" << endl;
    } else{
        throw FileCloseException(filename);
    }
};

FILE_STATUS printFileContent(string& filename) {
    try {
        openFile(filename);
        readFile(filename);
        parseFile(filename);
        closeFile(filename);
    } catch (FileException e) {
        cout << e.what() << endl;
        return e.code();
    }

    return FILE_STATUS_SUCCESS;
};


int main() {
    string filename;

    cout << "File Reading function" << endl;
    cout << "Please Enter File Name: " << ends;

    cin >> filename;

    FILE_STATUS fileRead = printFileContent(filename);

    switch (fileRead) {
        case FILE_STATUS_SUCCESS:
            cout << "Process Success" << endl;
            break;
        case FILE_STATUS_ERROR_READ_FILE:
        case FILE_STATUS_ERROR_NOT_FOUND:
        case FILE_STATUS_ERROR_PARSE_FILE:
        case FILE_STATUS_ERROR_CLOSE_FILE:
            cout << "Process Failed" << endl;
            break;
        default:
            break;
    }
}