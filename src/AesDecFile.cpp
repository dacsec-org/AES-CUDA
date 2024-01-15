#include "cryptlib.h"
#include "rijndael.h"
#include "modes.h"
#include "files.h"
#include "osrng.h"
#include "hex.h"
#include <iostream>
#include <string>

int main()
{
    // Get the AES key from the user
    std::string key;
    std::cout << "Enter the AES key (in hexadecimal format): ";
    std::cin >> key;

    // Convert the key from hexadecimal to byte array
    CryptoPP::HexDecoder decoder;
    decoder.Put((byte*)key.data(), key.size());
    decoder.MessageEnd();

    byte aesKey[CryptoPP::AES::DEFAULT_KEYLENGTH];
    decoder.Get(aesKey, sizeof(aesKey));

    // Generate a random initialization vector (IV)
    byte iv[CryptoPP::AES::BLOCKSIZE];
    CryptoPP::AutoSeededRandomPool rng;
    rng.GenerateBlock(iv, sizeof(iv));

    // Create an AES decryption object with the user-provided key and IV
    CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption aesDecryption(aesKey, sizeof(aesKey), iv);

    // Open the input file
    std::ifstream inputFile("output.enc", std::ios::binary);
    if (!inputFile)
    {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }

    // Open the output file
    std::ofstream outputFile("output.txt", std::ios::binary);
    if (!outputFile)
    {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    // Decrypt the input file and write the decrypted data to the output file
    CryptoPP::FileSource(inputFile, true,
        new CryptoPP::StreamTransformationFilter(aesDecryption,
            new CryptoPP::FileSink(outputFile)
        )
    );

    // Close the files
    inputFile.close();
    outputFile.close();

    std::cout << "File decrypted successfully" << std::endl;

    return 0;
}