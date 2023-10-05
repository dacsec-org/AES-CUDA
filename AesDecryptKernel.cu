#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <memory>
#include <chrono>
#include <filesystem>
// macro for error checking
#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

// function for error checking
void cudaCheck(cudaError_t result, [[maybe_unused]] const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at line " << line << ": " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

class AesDecryptor {
public:
    explicit AesDecryptor(size_t dataSize) : dataSize_(dataSize) {
        // Get CUDA device properties
        cudaDeviceProp deviceProps{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
        // Allocate pageable memory on host for input
        std::make_unique<uint8_t[]>(dataSize_);
        // Allocate regular memory on host for output
        output_host_ = std::make_unique<uint8_t[]>(dataSize_);
        // Allocate memory on GPU for input, output, and key
        CUDA_CHECK(cudaMalloc((void**)&input_gpu_, dataSize_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&output_gpu_, dataSize_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&key_gpu_, 176 * sizeof(uint8_t)));
    }

    ~AesDecryptor() {
        // Free GPU memory
        CUDA_CHECK(cudaFree(input_gpu_));
        CUDA_CHECK(cudaFree(output_gpu_));
        CUDA_CHECK(cudaFree(key_gpu_));
    }

    void Decrypt(const uint8_t* input, uint8_t* output, const uint8_t* key) {
        // Copy input and key from host to device
        CUDA_CHECK(cudaMemcpy(input_gpu_, input, dataSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(key_gpu_, key, 176, cudaMemcpyHostToDevice));
        // Set block and grid dimensions based on data size
        dim3 blockDims(numThreadsPerBlock_, 1, 1);
        dim3 gridDims((dataSize_ + numThreadsPerBlock_ - 1) / numThreadsPerBlock_, 1, 1);
        // Launch AES decryption kernel
        aesDecryptKernel<<<gridDims, blockDims>>>(input_gpu_, output_gpu_, key_gpu_, dataSize_);
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        // Copy output from GPU to host
        CUDA_CHECK(cudaMemcpy(output_host_.get(), output_gpu_, dataSize_, cudaMemcpyDeviceToHost));
        // Copy output from host to output buffer
        std::memcpy(output, output_host_.get(), dataSize_);
    }

private:
    size_t dataSize_;
    int numThreadsPerBlock_{256};
    std::unique_ptr<uint8_t[]> input_host_;
    std::unique_ptr<uint8_t[]> output_host_;
    uint8_t* input_gpu_{};
    uint8_t* output_gpu_{};
    uint8_t* key_gpu_{};

    // Inverse SubBytes table
    static const uint8_t invSubBytesTable[16][16];
    // Inverse MixColumns matrix
    static const uint8_t invMixColumnsMatrix[4][4];

    // Function to multiply two 8-bit values
    static __host__ __device__ uint8_t multiply(uint8_t a, uint8_t b) {
        uint8_t p = 0x00;
        for (int i = 0; i < 8; i++) {
            if (b & 0x01) {
                p ^= a;
            }
            uint8_t msb = a & 0x80;
            a <<= 1;
            if (msb) {
                a ^= 0x1B;  // XOR with the irreducible polynomial x^8 + x^4 + x^3 + x + 1
            }
            b >>= 1;
        }
        return p;
    }

    // inverseShiftRows (shift to the left)
    static __host__ __device__ void inverseShiftRows(uint8_t *state) {
        uint8_t temp;
        // Row 1
        temp = state[13];
        state[13] = state[9];
        state[9] = state[5];
        state[5] = state[1];
        state[1] = temp;
        // Row 2
        temp = state[2];
        state[2] = state[10];
        state[10] = temp;
        temp = state[6];
        state[6] = state[14];
        state[14] = temp;
        // Row 3
        temp = state[3];
        state[3] = state[7];
        state[7] = state[11];
        state[11] = state[15];
        state[15] = temp;
    }

    // inverseSubBytes
    static __host__ __device__ void inverseSubBytes(uint8_t *state) {
        for (int i = 0; i < 16; i++) {
            int row = (i >> 2) & 0x03;
            int col = i & 0x03;
            state[i] = invSubBytesTable[row][col];
        }
    }

    // addRoundKey
    static __host__ __device__ void addRoundKey(uint8_t *state, const uint8_t *key, int round) {
        for (int i = 0; i < 16; i++) {
            state[i] ^= key[round * 16 + i];
        }
    }

    // inverseMixColumns
    static __host__ __device__ void inverseMixColumns(uint8_t *state) {
        for (int col = 0; col < 4; col++) {
            uint8_t s0 = state[col];
            uint8_t s1 = state[col + 4];
            uint8_t s2 = state[col + 8];
            uint8_t s3 = state[col + 12];
            state[col] = multiply(s0, invMixColumnsMatrix[0][col]) ^
                         multiply(s1, invMixColumnsMatrix[1][col]) ^
                         multiply(s2, invMixColumnsMatrix[2][col]) ^
                         multiply(s3, invMixColumnsMatrix[3][col]);
            state[col + 4] = multiply(s0, invMixColumnsMatrix[0][(col + 1) % 4]) ^
                             multiply(s1, invMixColumnsMatrix[1][(col + 1) % 4]) ^
                             multiply(s2, invMixColumnsMatrix[2][(col + 1) % 4]) ^
                             multiply(s3, invMixColumnsMatrix[3][(col + 1) % 4]);
            state[col + 8] = multiply(s0, invMixColumnsMatrix[0][(col + 2) % 4]) ^
                             multiply(s1, invMixColumnsMatrix[1][(col + 2) % 4]) ^
                             multiply(s2, invMixColumnsMatrix[2][(col + 2) % 4]) ^
                             multiply(s3, invMixColumnsMatrix[3][(col + 2) % 4]);
            state[col + 12] = multiply(s0, invMixColumnsMatrix[0][(col + 3) % 4]) ^
                              multiply(s1, invMixColumnsMatrix[1][(col + 3) % 4]) ^
                              multiply(s2, invMixColumnsMatrix[2][(col + 3) % 4]) ^
                              multiply(s3, invMixColumnsMatrix[3][(col + 3) % 4]);
        }
    }

    // copyState
    static __host__ __device__ void copyState(const uint8_t *input, uint8_t *output) {
        for (int i = 0; i < 16; i++) {
            output[i] = input[i];
        }
    }

    // CUDA kernel for AES decryption
    static __global__ void
    aesDecryptKernel(const uint8_t *input, uint8_t *output, const uint8_t *key, size_t dataSize) {
        // Variable declarations
        size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < dataSize / 16) {
            size_t offset = idx * 16;
            uint8_t state[16];
            // Copy input to state
            copyState(&input[offset], state);
            // AES decryption algorithm
            for (int round = 10; round >= 1; round--) {
                // ShiftRows and Inverse SubBytes
                inverseShiftRows(state);
                inverseSubBytes(state);
                // AddRoundKey
                addRoundKey(state, key, round);
                // Inverse MixColumns
                inverseMixColumns(state);
            }
            // Inverse ShiftRows and Inverse SubBytes
            inverseShiftRows(state);
            inverseSubBytes(state);
            // AddRoundKey for round 0
            addRoundKey(state, key, 0);
            // Copy state to output
            copyState(state, &output[offset]);
        }
    }
};

// Initialize constant arrays
const uint8_t AesDecryptor::invSubBytesTable[16][16] = {
    {0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb},
    {0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34,0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb},
    {0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e},
    {0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25},
    {0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92},
    {0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84},
    {0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06},
    {0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b},
    {0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73},
    {0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e},
    {0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b},
    {0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4},
    {0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f},
    {0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef},
    {0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61},
    {0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d}
};

const uint8_t AesDecryptor::invMixColumnsMatrix[4][4] = {
    {0x0e, 0x0b, 0x0d, 0x09},
    {0x09, 0x0e, 0x0b, 0x0d},
    {0x0d, 0x09, 0x0e, 0x0b},
    {0x0b, 0x0d, 0x09, 0x0e}
};

// requests, and decodes the users base64 encoded Key
std::vector<uint8_t> userKey() {
    std::string base64Key;
    std::cout << "Enter base64 encoded key (32 bytes '256-bit'): ";
    std::cin >> base64Key;
    std::vector<uint8_t> decodedKey;
    decodedKey.reserve(32);
    const std::string base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (char c : base64Key) {
        if (c == '=') {
            break;
        }
        auto it = std::find(base64Chars.begin(), base64Chars.end(), c);
        if (it != base64Chars.end()) {
            auto index = static_cast<uint8_t>(std::distance(base64Chars.begin(), it));
            decodedKey.push_back(index);
        }
    }
    if (decodedKey.size() != 32) {
        std::cerr << "Invalid key size." << std::endl;
        exit(1);
    }
    return decodedKey;
}

// Requests the input location of the file to be decrypted and retrieves file size
std::pair<std::string, std::vector<uint8_t>> inputLocation() {
    std::string filePath;
    std::cout << "Enter the complete path of the file to be decrypted (e.g., /home/username/path/file.enc): ";
    std::getline(std::cin, filePath);

    // Check if the file exists
    if (!std::filesystem::exists(filePath)) {
        std::cout << "Error: File does not exist." << std::endl;
        return { "", {} };
    }

    // Check if the file has the '.enc' extension
    std::string extension = std::filesystem::path(filePath).extension().string();
    if (extension != ".enc") {
        std::cout << "Error: Invalid file extension. Only '*.enc' files are allowed." << std::endl;
        return { "", {} };
    }

    // Open the file to retrieve file size and check if it is readable
    std::ifstream inputFile(filePath, std::ios::binary);
    if (!inputFile) {
        std::cerr << "Failed to open input file." << std::endl;
        return { "", {} };
    }

    // Get file size
    inputFile.seekg(0, std::ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);

    // Read file contents into vector
    std::vector<uint8_t> input(fileSize);
    inputFile.read(reinterpret_cast<char*>(input.data()), input.size());
    inputFile.close();

    return { filePath, input };
}

// Output location of the decrypted file (defaults to the user's home directory)
std::string outputLocation(const std::string& inputFilePath) {
    std::string homeDir = std::getenv("HOME");
    std::string fileName = std::filesystem::path(inputFilePath).filename().string();

    // Remove the '.enc' extension from the file name
    size_t lastDotIndex = fileName.find_last_of('.');
    if (lastDotIndex != std::string::npos) {
        fileName = fileName.substr(0, lastDotIndex);
    }

    std::string decryptedFilePath = homeDir + "/" + fileName;
    return decryptedFilePath;
}

int main() {
    std::pair<std::string, std::vector<uint8_t>> fileData = inputLocation();
    std::string filePath = fileData.first;
    std::vector<uint8_t> input = fileData.second;

    if (filePath.empty() || input.empty()) {
        return 1; // Exit if file input is invalid
    }

    std::vector<uint8_t> decodedKey = userKey();
    size_t dataSize = input.size();
    AesDecryptor decryptor(dataSize);
    std::vector<uint8_t> output(dataSize);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    decryptor.Decrypt(input.data(), output.data(), decodedKey.data());
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Decryption time: " << duration << " ms" << std::endl;

    std::string decryptedFilePath = outputLocation(filePath);
    std::ofstream outputFile(decryptedFilePath, std::ios::binary);
    if (outputFile) {
        outputFile.write(reinterpret_cast<char*>(output.data()), output.size());
        outputFile.close();
        std::cout << "Decrypted file written to: " << decryptedFilePath << std::endl;
    } else {
        std::cerr << "Failed to write decrypted file." << std::endl;
        return 1;
    }

    return 0;
}