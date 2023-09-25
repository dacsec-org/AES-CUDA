#include <cuda_runtime.h>
#include <iostream>
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(result) << std::endl; \
        exit(1); \
    } \
} while(0)
class AesDecryptor {
public:
    AesDecryptor() {
        // Get CUDA device properties
        cudaDeviceProp deviceProps{};
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
        // Get available global memory
        size_t totalGlobalMem, freeGlobalMem;
        CUDA_CHECK(cudaMemGetInfo(&freeGlobalMem, &totalGlobalMem));
        // Calculate data size based on available memory
        dataSize_ = freeGlobalMem * 0.8;
        // Calculate number of blocks and threads per block
        numBlocks_ = (dataSize_ + 15) / 16;
        numThreadsPerBlock_ = 256;
        // Allocate memory on GPU for input, output, and key
        CUDA_CHECK(cudaMalloc((void**)&input_gpu_, dataSize_));
        CUDA_CHECK(cudaMalloc((void**)&output_gpu_, dataSize_));
        CUDA_CHECK(cudaMalloc((void**)&key_gpu_, 176));
    }
    ~AesDecryptor() {
        // Free GPU memory
        CUDA_CHECK(cudaFree(input_gpu_));
        CUDA_CHECK(cudaFree(output_gpu_));
        CUDA_CHECK(cudaFree(key_gpu_));
    }
    void Decrypt(const uint8_t* input, uint8_t* output, const uint8_t* key) {
        // Copy input and key from host to GPU
        CUDA_CHECK(cudaMemcpy(input_gpu_, input, dataSize_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(key_gpu_, key, 176, cudaMemcpyHostToDevice));
        // Set block and grid dimensions
        dim3 blockDims(numThreadsPerBlock_, 1, 1);
        dim3 gridDims(numBlocks_, 1, 1);
        // Launch AES decryption kernel
        aesDecryptKernel<<<gridDims, blockDims>>>(input_gpu_, output_gpu_, key_gpu_);
        // Check for kernel launch errors
        CUDA_CHECK(cudaGetLastError());
        // Copy output from GPU to host
        CUDA_CHECK(cudaMemcpy(output, output_gpu_, dataSize_, cudaMemcpyDeviceToHost));
    }
    [[nodiscard]] size_t getDataSize() const {
        return dataSize_;
    }
private:
    size_t dataSize_;
    int numBlocks_;
    int numThreadsPerBlock_;
    uint8_t* input_gpu_{};
    uint8_t* output_gpu_{};
    uint8_t* key_gpu_{};
    // Inverse SubBytes table
    static __device__ __constant__ uint8_t invSubBytesTable[16][16];
    // Inverse MixColumns matrix
    static __device__ __constant__ uint8_t invMixColumnsMatrix[4][4];
    // Function to multiply two 8-bit values
    static __device__ uint8_t multiply(uint8_t a, uint8_t b) {
        uint8_t result = 0;
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
    // CUDA kernel for AES decryption
    static __global__ void aesDecryptKernel(const uint8_t* input, uint8_t* output, const uint8_t* key) {
        // Variable declarations
        unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int round;
        uint8_t state[16];
        // Copy input to state
        for (int i = 0; i < 16; i++) {
            state[i] = input[idx * 16 + i];
        }
        // AddRoundKey for round 10
        for (int i = 0; i < 16; i++) {
            state[i] ^= key[160 + i];
        }
        // AES decryption algorithm
        for (round = 9; round >= 1; round--) {
            // Shift the 13'th Row the left
            uint8_t temp = state[13];
            state[13] = state[9];
            state[9] = state[5];
            state[5] = state[1];
            state[1] = temp;
            // Shift the 10'th Row the left
            temp = state[10];
            state[10] = state[2];
            state[2] = temp;
            // Shift the 7'th Row the right
            temp = state[7];
            state[7] = state[11];
            state[11] = state[15];
            state[15] = state[3];
            state[3] = temp;
            // Inverse SubBytes
            for (unsigned char& i : state) {
                int row = (i >> 4) & 0x0F;
                int col = i & 0x0F;
                i = invSubBytesTable[row][col];
            }
            // AddRoundKey
            for (int i = 0; i < 16; i++) {
                state[i] ^= key[round * 16 + i];
            }
            // Inverse MixColumns
            for (int col = 0; col < 4; col++) {
                uint8_t s0 = state[col];
                uint8_t s1 = state[col + 4];
                uint8_t s2 = state[col + 8];
                uint8_t s3 = state[col + 12];
                state[col] = multiply(s0, invMixColumnsMatrix[0][0]) ^
                    multiply(s1, invMixColumnsMatrix[0][1]) ^
                    multiply(s2, invMixColumnsMatrix[0][2]) ^
                    multiply(s3, invMixColumnsMatrix[0][3]);
                state[col + 4] = multiply(s0, invMixColumnsMatrix[1][0]) ^
                    multiply(s1, invMixColumnsMatrix[1][1]) ^
                    multiply(s2, invMixColumnsMatrix[1][2]) ^
                    multiply(s3, invMixColumnsMatrix[1][3]);
                state[col + 8] = multiply(s0, invMixColumnsMatrix[2][0]) ^
                    multiply(s1, invMixColumnsMatrix[2][1]) ^
                    multiply(s2, invMixColumnsMatrix[2][2]) ^
                    multiply(s3, invMixColumnsMatrix[2][3]);
                state[col + 12] = multiply(s0, invMixColumnsMatrix[3][0]) ^
                    multiply(s1, invMixColumnsMatrix[3][1]) ^
                    multiply(s2, invMixColumnsMatrix[3][2]) ^
                    multiply(s3, invMixColumnsMatrix[3][3]);
            }
            // Inverse SubBytes
            for (unsigned char& i : state) {
                int row = (i >> 4) & 0x0F;
                int col = i & 0x0F;
                i = invSubBytesTable[row][col];
            }
        }
        // After Inverse mix columns are multiplied, Inverse ShiftRows again to the left
        uint8_t temp = state[13];
        state[13] = state[9];
        state[9] = state[5];
        state[5] = state[1];
        state[1] = temp;
        temp = state[10];
        state[10] = state[2];
        state[2] = temp;
        temp = state[7];
        state[7] = state[11];
        state[11] = state[15];
        state[15] = state[3];
        state[3] = temp;
        // Inverse SubBytes
        for (unsigned char& i : state) {
            int row = (i >> 4) & 0x0F;
            int col = i & 0x0F;
            i = invSubBytesTable[row][col];
        }
        // AddRoundKey for round 0
        for (int i = 0; i < 16; i++) {
            state[i] ^= key[i];
        }
        // Copy state to output
        for (int i = 0; i < 16; i++) {
            output[idx * 16 + i] = state[i];
        }
    }
};
// Initialize constant arrays
__device__ __constant__ uint8_t AesDecryptor::invSubBytesTable[16][16] = {
        {0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf,
                0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb},
        {0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34,
                0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb},
        {0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee,
                0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e},
        {0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76,
                0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25},
        {0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4,
                0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92},
        {0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e,
                0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84},
        {0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7,
                0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06},
        {0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1,
                0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b},
        {0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97,
                0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73},
        {0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2,
                0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e},
        {0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f,
                0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b},
        {0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a,
                0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4},
        {0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1,
                0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f},
        {0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d,
                0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef},
        {0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8,
                0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61},
        {0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1,
                0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d}
};
__device__ __constant__ uint8_t AesDecryptor::invMixColumnsMatrix[4][4] = {
        {0x0e, 0x0b, 0x0d, 0x09},
        {0x09, 0x0e, 0x0b, 0x0d},
        {0x0d, 0x09, 0x0e, 0x0b},
        {0x0b, 0x0d, 0x09, 0x0e}
};
// Function to get input from user
void getInput(uint8_t* input) {
    std::cout << "Enter input data (16 bytes in hexadecimal): ";
    for (int i = 0; i < 16; i++) {
        std::cin >> std::hex >> input[i];
    }
}
// Function to get key from user
void getKey(uint8_t* key) {
    std::cout << "Enter key data (176 bytes in hexadecimal): ";
    for (int i = 0; i < 176; i++) {
        std::cin >> std::hex >> key[i];
    }
}
int main() {
    // Test the AesDecryptor class
    AesDecryptor decryptor;
    // Test input data
    uint8_t input[16];
    getInput(input);
    // Test key data
    uint8_t key[176];
    getKey(key);
    // Perform decryption
    uint8_t output[16];
    decryptor.Decrypt(input, output, key);
    // Print the decrypted output
    for (unsigned char i : output) {
        std::cout << std::hex << static_cast<int>(i) << " ";
    }
    std::cout << std::endl;
    // Get the data size
    size_t dataSize = decryptor.getDataSize();
    std::cout << "Data size: " << dataSize << std::endl;
    return 0;
}