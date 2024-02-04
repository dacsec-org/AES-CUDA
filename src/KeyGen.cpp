#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cryptopp/osrng.h>
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/files.h>
#include <cryptopp/base64.h>
#include <cryptopp/pwdbased.h>
#include <cryptopp/sha.h>

typedef unsigned char byte;

class KeyGen {
public:
    void generateKey() {
        std::string password;
        std::cout << "Please enter a strong password for the key: ";
        std::getline(std::cin, password);

        // Check if key file already exists
        std::string homeDir = getenv("HOME");
        std::string kryptDir = homeDir + "/KryptFile";
        std::string keyFile = kryptDir + "/key.bin";
        if (std::filesystem::exists(keyFile)) {
            std::cout << "Key file already exists. Aborting key generation." << std::endl;
            return;
        }

        CryptoPP::AutoSeededRandomPool rng;
        CryptoPP::SecByteBlock key(CryptoPP::AES::DEFAULT_KEYLENGTH);
        CryptoPP::SecByteBlock iv(CryptoPP::AES::BLOCKSIZE);

        // Generate random key and IV
        rng.GenerateBlock(key, key.size());
        rng.GenerateBlock(iv, iv.size());

        // Set password as the encryption key
        CryptoPP::SecByteBlock derivedKey(CryptoPP::AES::DEFAULT_KEYLENGTH);
        CryptoPP::PKCS5_PBKDF2_HMAC<CryptoPP::SHA256> pbkdf;
        pbkdf.DeriveKey(derivedKey, derivedKey.size(), 0x00, (byte*)password.data(), password.size(), NULL, 0, 1000);

        // Encrypt the key using derived key and IV
        CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption encryption(derivedKey, derivedKey.size(), iv);
        CryptoPP::ArraySink encryptedKey(key, key.size());
        CryptoPP::ArraySource(key, key.size(), true, new CryptoPP::StreamTransformationFilter(encryption, new CryptoPP::Redirector(encryptedKey)));

        // Create the KryptFile directory if it doesn't exist
        if (!std::filesystem::exists(kryptDir)) {
            std::filesystem::create_directory(kryptDir);
        }

        // Write the encrypted key to the file
        std::ofstream outputFile(keyFile, std::ios::binary);
        if (!outputFile) {
            std::cout << "Failed to open key file for writing." << std::endl;
            return;
        }
        outputFile.write(reinterpret_cast<const char*>(encryptedKey.GetArray()), encryptedKey.SizeInBytes());
        outputFile.close();

        // Securely wipe the memory
        derivedKey.Assign(derivedKey.size(), 0x00);
        password.assign(password.size(), '0');
        std::cout << "Key generated and saved to " << keyFile << std::endl;
    }
};

int main() {
    KeyGen keyGen;
    keyGen.generateKey();
    return 0;
}