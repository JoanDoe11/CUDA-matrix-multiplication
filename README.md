# CUDA-matrix-multiplication
Solution of matrix multiplication using PyCUDA

# CUDA
[src = nvidia]
CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on GPUs. 

In GPU-accelerated applications, the sequential part of the workload runs on the CPU - which is optimized for single-threaded performance - while the compute intensive portion of the application runs on thousands of GPU cores in parallel. 

The CUDA platform is a software layer that gives direct access to the GPU's virtual instruction set and parallel computational elements, for the execution of kernels. It is accessible to software developers through CUDA-accelerated libraries, compiler directives and extensions to industry-standard programming languages. 

A typical CUDA program has code for both the GPU(device) and the CPU(host). The NVCC compiler processes a CUDA program, and separates the host code from the device code. The device code is then further compiled by the NVCC and executed on the GPU.
