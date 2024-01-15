#include <iostream>
#include <fstream>
#include <string>
#include <cryptopp/osrng.h>
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/files.h>
#include <cryptopp/base64.h>

/*
KeyGen is a class  that uses the Crypto++ library to generate a password-protected 256-bit AES key and save it to the user's home directory in the "KryptFile" directory.
*/
class KeyGen {
public:
    void generateKey() {
        std::string password;
        std::cout << "Please nter a strong password for the key: ";
        std::getline(std::cin, password);

        // Check if key file already exists
        std::string homeDir = getenv("HOME");
        std::string kryptDir = homeDir + "/KryptFile";
        std::string keyFile = kryptDir + "/key.bin";

        if (fileExists(keyFile)) {
            std::cout << "Key file already exists. Aborting key generation." << std::endl;
            return;
        }

        CryptoPP::AutoSeededRandomPool rng;
        byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
        byte iv[CryptoPP::AES::BLOCKSIZE];

        // Generate random key and IV
        rng.GenerateBlock(key, sizeof(key));
        rng.GenerateBlock(iv, sizeof(iv));

        // Set password as the encryption key
        CryptoPP::SecByteBlock derivedKey(CryptoPP::AES::DEFAULT_KEYLENGTH);
        CryptoPP::PKCS5_PBKDF2_HMAC<CryptoPP::SHA256> pbkdf;
        pbkdf.DeriveKey(derivedKey, derivedKey.size(), 0x00, (byte*)password.data(), password.size(), NULL, 0, 1000);

        // Encrypt the key using derived key and IV
        CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption encryption(derivedKey, derivedKey.size(), iv);
        CryptoPP::ArraySink encryptedKey(key, sizeof(key));
        CryptoPP::ArraySource(key, sizeof(key), true, new CryptoPP::StreamTransformationFilter(encryption, new CryptoPP::Redirector(encryptedKey)));

        // Create the KryptFile directory if it doesn't exist
        if (!createDirectory(kryptDir)) {
            std::cout << "Failed to create KryptFile directory." << std::endl;
            return;
        }

        // Write the encrypted key to the file
        std::ofstream outputFile(keyFile, std::ios::binary);
        if (!outputFile) {
            std::cout << "Failed to open key file for writing." << std::endl;
            return;
        }

        CryptoPP::Base64Encoder encoder(new CryptoPP::FileSink(outputFile));
        encoder.Put(encryptedKey, sizeof(key));
        encoder.MessageEnd();

        std::cout << "Key generated and saved to " << keyFile << std::endl;
    }

private:
    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }

    bool createDirectory(const std::string& directory) {
        int result = mkdir(directory.c_str(), 0700);
        if (result == 0) {
            return true;
        }
        else if (errno == EEXIST) {
            return true;
        }
        else {
            return false;
        }
    }
};

int main() {
    KeyGen keyGen;
    keyGen.generateKey();

    return 0;
}