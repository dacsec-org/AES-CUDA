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
    // Generate a random password protected 32 =-bit key
    CryptoPP::AutoSeededRandomPool rng;
    byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
    rng.GenerateBlock(key, sizeof(key));
    // TODO: save the key to the users home dir 

    // Generate a random initialization vector (IV)
    byte iv[CryptoPP::AES::BLOCKSIZE];
    rng.GenerateBlock(iv, sizeof(iv));

    // Create an AES encryption object with the generated key and IV
    CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption aesEncryption(key, sizeof(key), iv);

    // Open the input file
    std::ifstream inputFile("input.txt", std::ios::binary);
    if (!inputFile)
    {
        std::cerr << "Failed to open input file" << std::endl;
        return 1;
    }

    // Open the output file
    std::ofstream outputFile("output.enc", std::ios::binary);
    if (!outputFile)
    {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    // Encrypt the input file and write the encrypted data to the output file
    CryptoPP::FileSource(inputFile, true,
        new CryptoPP::StreamTransformationFilter(aesEncryption,
            new CryptoPP::FileSink(outputFile)
        )
    );

    // Close the files
    inputFile.close();
    outputFile.close();

    std::cout << "File encrypted successfully" << std::endl;

    return 0;
}