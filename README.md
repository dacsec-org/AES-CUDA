# AES-CUDA-Kernels

This repository provides a GPU-accelerated solution for AES (Advanced Encryption Standard) encryption and decryption processes. By utilizing the power of the GPU, the 'aes_encrypt_kernel.cu' and 'aes_decrypt_kernel.cu' files offer significant performance improvements over traditional CPU-based implementations.

## Features

- GPU-accelerated AES encryption and decryption
- Improved performance and speed
- Easy integration into existing projects
- Support for 256-bit AES keys
- Efficient memory management for optimal GPU utilization

[//]: # (## Installation)

[//]: # ()
[//]: # (To use this GPU-accelerated AES encryption/decryption solution, follow these steps:)

[//]: # ()
[//]: # (1. Clone this repository to your local machine.)

[//]: # (2. Ensure you have the necessary CUDA toolkit installed.)

[//]: # (3. Compile the 'aes_encrypt_kernel.cu' and 'aes_decrypt_kernel.cu' files using your preferred CUDA compiler.)

[//]: # (4. Link the generated object files with your project.)

[//]: # (5. Make sure to include the necessary header files in your code.)

[//]: # (## Usage)

[//]: # ()
[//]: # (To encrypt data using the GPU-accelerated AES encryption:)

[//]: # (#include "aes_encrypt_kernel.cuh")

[//]: # ()
[//]: # (// Initialize AES key and input data)

[//]: # (unsigned char key[32] = { /* AES key */ };)

[//]: # (unsigned char plaintext[16] = { /* Input data */ };)

[//]: # ()
[//]: # (// Encrypt the data using the GPU-accelerated AES encryption kernel)

[//]: # (aes_encrypt_gpu&#40;key, plaintext&#41;;)

[//]: # ()
[//]: # (// The encrypted data is now stored in the 'plaintext' array)

[//]: # (To decrypt data using the GPU-accelerated AES decryption:)

[//]: # (#include "aes_decrypt_kernel.cuh")

[//]: # ()
[//]: # (// Initialize AES key and encrypted data)

[//]: # (unsigned char key[32] = { /* AES key */ };)

[//]: # (unsigned char ciphertext[16] = { /* Encrypted data */ };)

[//]: # ()
[//]: # (// Decrypt the data using the GPU-accelerated AES decryption kernel)

[//]: # (aes_decrypt_gpu&#40;key, ciphertext&#41;;)

[//]: # ()
[//]: # (// The decrypted data is now stored in the 'ciphertext' array)

## Performance

The GPU acceleration provided by this repository offers significant performance improvements compared to traditional CPU-based AES encryption/decryption. The parallel processing capabilities of the GPU enable faster and more efficient encryption and decryption of data, making it ideal for applications that require high-speed cryptographic operations.

## Contributing

Contributions to this repository are welcome. If you have any ideas, improvements, or bug fixes, please feel free to submit a pull request. Together, we can enhance the GPU-accelerated AES encryption/decryption solution and make it even more powerful.

[//]: # (## License)

[//]: # ()
[//]: # (This repository is licensed under the [MIT License]&#40;LICENSE&#41;, allowing you to use, modify, and distribute the code freely.)

## Acknowledgments

We would like to express our gratitude to the CUDA development team for providing the powerful tools and frameworks that made this GPU-accelerated AES encryption/decryption solution possible.

[//]: # (## Contact)

[//]: # ()
[//]: # (If you have any questions, suggestions, or feedback, please don't hesitate to contact us at [email protected] We would be happy to assist you.)

Happy GPU-accelerated AES encryption/decryption!