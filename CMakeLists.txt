cmake_minimum_required(VERSION 3.25)
project(AES_CUDA_Kernels CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(AES_CUDA_Kernels AesEncryptKernel.cu
        AesDecryptKernel.cu
        AesDecFile.cpp
        AesEncFile.cpp
        aes_decrypt.cuh
        aes_encrypt.cuh)

set_target_properties(AES_CUDA_Kernels PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
