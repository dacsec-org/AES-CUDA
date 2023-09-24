/*
 * The `aesDecryptKernel` is a CUDA kernel that performs AES decryption on the input data.
 * It takes three arguments: `const uint8_t* input`, `uint8_t* output`, and `const uint8_t* key`.
 * The kernel is launched with a grid of `numBlocks` blocks and `blockDims` threads per block.
 * The kernel implements the AES decryption algorithm by applying the inverse transformations to the input data.
 * It consists of multiple rounds, each performing the inverse of SubBytes, ShiftRows, MixColumns, and AddRoundKey operations.
 * The inverse SubBytes operation replaces each byte with a corresponding value from the inverse SubBytes table.
 * The inverse ShiftRows operation shifts the bytes in each row to the right.
 * The inverse MixColumns operation performs a matrix multiplication on each column of the state using the inverse MixColumns matrix.
 * The inverse AddRoundKey operation XORs the state with the round key.
 * After the last round, the inverse SubBytes and inverse ShiftRows operations are performed again, followed by the final inverse AddRoundKey operation.
 * The resulting state is then copied to the output array.
 */
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

// Inverse SubBytes table
__device__ __constant__ uint8_t invSubBytesTable[16][16] = {
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

// Inverse MixColumns matrix
__device__ __constant__ uint8_t invMixColumnsMatrix[4][4] = {
        {0x0e, 0x0b, 0x0d, 0x09},
        {0x09, 0x0e, 0x0b, 0x0d},
        {0x0d, 0x09, 0x0e, 0x0b},
        {0x0b, 0x0d, 0x09, 0x0e}
};

// Function to multiply two 8-bit values
__device__ uint8_t multiply(uint8_t a, uint8_t b) {
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
__global__ void aesDecryptKernel(const uint8_t* input, uint8_t* output, const uint8_t* key)
{
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
        for (unsigned char & i : state) {
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
        for (unsigned char & i : state) {
            int row = (i >> 4) & 0x0F;
            int col = i & 0x0F;
            i = invSubBytesTable[row][col];
        }
    }

    // Inverse ShiftRows
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
    for (unsigned char & i : state) {
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

int main()
{
    // Get CUDA device properties
    cudaDeviceProp deviceProps{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));

    // Get available global memory
    size_t totalGlobalMem, freeGlobalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeGlobalMem, &totalGlobalMem));

    // Calculate data size based on available memory
    const size_t dataSize = freeGlobalMem * 0.8;

    // Calculate number of blocks and threads per block
    const int numBlocks = (dataSize + 15) / 16;
    const int numThreadsPerBlock = 256;

    // Allocate memory on GPU for input, output, and key
    uint8_t* input_gpu;
    CUDA_CHECK(cudaMalloc((void**)&input_gpu, dataSize));
    uint8_t* output_gpu;
    CUDA_CHECK(cudaMalloc((void**)&output_gpu, dataSize));
    uint8_t* key_gpu;
    CUDA_CHECK(cudaMalloc((void**)&key_gpu, 176));

    // Set block and grid dimensions
    dim3 blockDims(numThreadsPerBlock, 1, 1);
    dim3 gridDims(numBlocks, 1, 1);

    // Launch AES decryption kernel
    aesDecryptKernel<<<gridDims, blockDims>>>(input_gpu, output_gpu, key_gpu);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Allocate host memory for output
    auto* output = new uint8_t[dataSize];

    // Copy output from GPU to host
    CUDA_CHECK(cudaMemcpy(output, output_gpu, dataSize, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(input_gpu));
    CUDA_CHECK(cudaFree(output_gpu));
    CUDA_CHECK(cudaFree(key_gpu));

    // Free host memory
    delete[] output;

    return 0;
}