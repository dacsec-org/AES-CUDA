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

// Class for AES encryption
class AesEncryptor {
public:
    explicit AesEncryptor(size_t dataSize) : dataSize_(dataSize) {
        // Get CUDA device properties
        // Get CUDA device properties
        cudaDeviceProp deviceProps{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
        // Allocate pageable memory on host for input
        std::make_unique<uint8_t[]>(dataSize_);
        // Allocate regular memory on host for output
        std::make_unique<uint8_t[]>(dataSize_);
        // Allocate memory on GPU for input, output, and key
        CUDA_CHECK(cudaMalloc((void**)&input_gpu_, dataSize_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&output_gpu_, dataSize_ * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc((void**)&key_gpu_, 176 * sizeof(uint8_t)));
    }

    ~AesEncryptor() {
        // Free GPU memory
        CUDA_CHECK(cudaFree(input_gpu_));
        CUDA_CHECK(cudaFree(output_gpu_));
        CUDA_CHECK(cudaFree(key_gpu_));
    }

    void Encrypt(const uint8_t* input, uint8_t* output, const uint8_t* key) {
        // Copy input and key from host to GPU
        CUDA_CHECK(cudaMemcpy(input_gpu_, input, dataSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(key_gpu_, key, 176, cudaMemcpyHostToDevice));
        // Set block and grid dimensions
        dim3 blockDims(numThreadsPerBlock_, 1, 1);
        dim3 gridDims((dataSize_ + numThreadsPerBlock_ - 1) / numThreadsPerBlock_, 1, 1);
        // Launch AES encryption kernel
        aesEncryptKernel<<<gridDims, blockDims>>>(input_gpu_, output_gpu_, key_gpu_);
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

    // SubBytes table
    static const uint8_t subBytesTable[16][16];
    // MixColumns matrix
    static const uint8_t mixColumnsMatrix[4][4];

    // Function to multiply two 8-bit values
    static __host__ __device__ uint8_t multiply(uint8_t a, uint8_t b) {
        uint8_t mask = 0x01;
        uint8_t p = 0x00;
        for (int i = 0; i < 8; i++) {
            if (b & mask) {
                p ^= a;
            }
            uint8_t msb = a & 0x80;
            a <<= 1;
            if (msb) {
                a ^= 0x1B;  // XOR with the irreducible polynomial x^8 + x^4 + x^3 + x + 1
            }
            mask <<= 1;
        }
        return p;
    }

    // CUDA kernel for AES encryption
    static __global__ void aesEncryptKernel(const uint8_t* input, uint8_t* output, const uint8_t* key) {
        // Variable declarations
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int round;
        uint8_t state[16];
        // Copy input to state
        copyInputToState(input, idx, state);
        // AddRoundKey for round 0
        addRoundKey(state, key, 0);
        // AES encryption algorithm
        for (round = 1; round <= 9; round++) {
            // SubBytes
            subBytes(state);
            // ShiftRows
            shiftRows(state);
            // MixColumns
            mixColumns(state);
            // AddRoundKey
            addRoundKey(state, key, round);
        }
        // SubBytes
        subBytes(state);
        // ShiftRows
        shiftRows(state);
        // AddRoundKey for round 10
        addRoundKey(state, key, 10);
        // Copy state to output
        copyStateToOutput(output, idx, state);
    }

    // Function to copy input to state
    static __host__ __device__ void copyInputToState(const uint8_t* input, unsigned int idx, uint8_t* state) {
        for (int i = 0; i < 16; i++) {
            state[i] = input[idx * 16 + i];
        }
    }

    // Function for SubBytes transformation
    static __host__ __device__ void subBytes(uint8_t* state) {
        for (int i = 0; i < 16; i++) {
            int row = (state[i] >> 4) & 0x0F;
            int col = state[i] & 0x0F;
            state[i] = subBytesTable[row][col];
        }
    }

    // Function for ShiftRows transformation
    static __host__ __device__ void shiftRows(uint8_t* state) {
        uint8_t temp;
        // Shift the 1st row to the right
        temp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = temp;
        // Shift the 2nd row to the right
        temp = state[2];
        state[2] = state[10];
        state[10] = temp;
        // Shift the 3rd row to the right
        temp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = temp;
    }

    // Function for MixColumns transformation
    static __host__ __device__ void mixColumns(uint8_t* state) {
        for (int col = 0; col < 4; col++) {
            uint8_t s0 = state[col];
            uint8_t s1 = state[col + 4];
            uint8_t s2 = state[col + 8];
            uint8_t s3 = state[col + 12];

            state[col] = multiply(s0, mixColumnsMatrix[0][0]) ^
                         multiply(s1, mixColumnsMatrix[0][1]) ^
                         multiply(s2, mixColumnsMatrix[0][2]) ^
                         multiply(s3, mixColumnsMatrix[0][3]);

            state[col + 4] = multiply(s0, mixColumnsMatrix[1][0]) ^
                             multiply(s1, mixColumnsMatrix[1][1]) ^
                             multiply(s2, mixColumnsMatrix[1][2]) ^
                             multiply(s3, mixColumnsMatrix[1][3]);

            state[col + 8] = multiply(s0, mixColumnsMatrix[2][0]) ^
                             multiply(s1, mixColumnsMatrix[2][1]) ^
                             multiply(s2, mixColumnsMatrix[2][2]) ^
                             multiply(s3, mixColumnsMatrix[2][3]);

            state[col + 12] = multiply(s0, mixColumnsMatrix[3][0]) ^
                              multiply(s1, mixColumnsMatrix[3][1]) ^
                              multiply(s2, mixColumnsMatrix[3][2]) ^
                              multiply(s3, mixColumnsMatrix[3][3]);
        }
    }

    // Function for AddRoundKey transformation
    static __host__ __device__ void addRoundKey(uint8_t* state, const uint8_t* key, int round) {
        for (int i = 0; i < 16; i++) {
            state[i] ^= key[round * 16 + i];
        }
    }

    // Function to copy state to output
    static __host__ __device__ void copyStateToOutput(uint8_t* output, unsigned int idx, const uint8_t* state) {
        for (int i = 0; i < 16; i++) {
            output[idx * 16 + i] = state[i];
        }
    }
};

// Initialize constant arrays
const uint8_t AesEncryptor::subBytesTable[16][16] = {
        {0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30,
                0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
        {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad,
                0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
        {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34,
                0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
        {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07,
                0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
        {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52,
                0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
        {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a,
                0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
        {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
                0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
        {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc,
                0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
        {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4,
                0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
        {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46,
                0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
        {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2,
                0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
        {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c,
                0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
        {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8,
                0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
        {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61,
                0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
        {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b,
                0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
        {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41,
                0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}
};

const uint8_t AesEncryptor::mixColumnsMatrix[4][4] = {
        {0x02, 0x03, 0x01, 0x01},
        {0x01, 0x02, 0x03, 0x01},
        {0x01, 0x01, 0x02, 0x03},
        {0x03, 0x01, 0x01, 0x02}
};

// Requests the base64 encoded key from the user and decodes it
std::vector<uint8_t> userKey() {
    std::string base64Key;
    std::cout << "Enter base64 encoded key (32 bytes '256-bit'): ";
    std::cin >> base64Key;
    std::vector<uint8_t> decodedKey;
    decodedKey.reserve(32);
    const std::string base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (char c: base64Key) {
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

// Requests the input location of the file to be encrypted and retrieves file size
std::pair<std::string, std::vector<uint8_t>> inputLocation() {
    std::string filePath;
    std::cout << "Enter the complete path of the file to be encrypted (e.g., /home/username/path/file): ";
    std::getline(std::cin, filePath);
    // Check if the file exists
    if (!std::filesystem::exists(filePath)) {
        std::cout << "Error: File does not exist." << std::endl;
        return { "", {} };
    }

    // Check if the file does not have the '.enc' extension
    std::string extension = std::filesystem::path(filePath).extension().string();
    if (extension == ".enc") {
        std::cout << "Error: Invalid file extension. Only non '*.enc' files are allowed." << std::endl;
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

// Output location of the encrypted file (defaults to the user's home directory)
std::string outputLocation(const std::string& inputFilePath) {
    std::string homeDir = std::getenv("HOME");
    std::string fileName = std::filesystem::path(inputFilePath).filename().string();
    std::string encryptedFilePath = homeDir + "/" + fileName + ".enc";
    return encryptedFilePath;
}

int main() {
    std::pair<std::string, std::vector<uint8_t>> fileData = inputLocation();
    std::string filePath = fileData.first;
    std::vector<uint8_t> input = fileData.second;
    if (filePath.empty() || input.empty()) {
        return 1; // Exit if file input is invalid
    }
    std::vector<uint8_t> key = userKey();
    size_t dataSize = input.size();
    AesEncryptor encryptor(dataSize);
    std::vector<uint8_t> output(dataSize);
    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    encryptor.Encrypt(input.data(), output.data(), key.data());
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Encryption time: " << duration << " ms" << std::endl;
    std::string encryptedFilePath = outputLocation(filePath);
    std::ofstream outputFile(encryptedFilePath, std::ios::binary);
    if (outputFile) {
        outputFile.write(reinterpret_cast<char*>(output.data()), output.size());
        outputFile.close();
        std::cout << "Encrypted file written to: " << encryptedFilePath << std::endl;
    } else {
        std::cerr << "Failed to write encrypted file." << std::endl;
        return 1;
    }
    return 0;
}