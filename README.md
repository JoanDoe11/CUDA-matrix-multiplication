# CUDA-matrix-multiplication
Solution of matrix multiplication using PyCUDA

# CUDA
[src = nvidia]
CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on GPUs. 

In GPU-accelerated applications, the sequential part of the workload runs on the CPU - which is optimized for single-threaded performance - while the compute intensive portion of the application runs on thousands of GPU cores in parallel. 

The CUDA platform is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of kernels. It is accessible to software developers through CUDA-accelerated libraries, compiler directives and extensions to industry-standard programming languages. 

A typical CUDA program has code for both the GPU(device) and the CPU(host). The NVCC compiler processes a CUDA program, and separates the host code from the device code. The device code is then further compiled by the NVCC and executed on the GPU.

The programmer has explicit control on the number of threads that he wants to launch. These threads collectively form a three-dimensional grid - threads are packed into bloks, and blocks are packed into grids. 

To execute a kernel (data parallel function) on the GPU, the programmer needs to allocate separate memory on the GPU. The host can access the device memory and transfer data to and from it, but not the other way around.

CUDA flow sequence:
1. allocating memory on device; data transfered from host to device memory. 
2. kernel is executed on the device; result is transferred back from device to host memory.
3. free up the allocated memory on the device; 

Threads in a grid execute the same kernel function. They have specific coordinates to distinguish themselves from eachother and identify the relevant portion of data to process. Execution resources are assigned to threads per block. Resources are organized into Streaming Multiprocessors. Multiple blocks of threads can be assigned to a single SM.
