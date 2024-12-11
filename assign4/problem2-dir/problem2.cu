 #include <cassert>
 #include <cstdlib>
 #include <cuda.h>
 #include <iostream>
 #include <numeric>
 #include <sys/time.h>
 #include <thrust/device_vector.h>
 #include <thrust/scan.h>

 using std::cerr;
 using std::cout;
 using std::endl;

 #define cudaCheckError(ans)                                                    \
   { gpuAssert((ans), __FILE__, __LINE__); }
 inline void gpuAssert(cudaError_t code, const char* file, int line,
                       bool abort = true) {
   if (code != cudaSuccess) {
     fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
             line);
     if (abort)
       exit(code);
   }
 }

 const uint64_t N = (1 << 20);

 __host__ void thrust_sum(const uint32_t* input, uint32_t* output) {
    thrust::device_vector<uint32_t> d_in(input, input + N);
    thrust::device_vector<uint32_t> d_out(N);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);  
    thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
    thrust::copy(d_out.begin(), d_out.end(), output);
    cudaEventRecord(end);
    cudaCheckError(cudaDeviceSynchronize());
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Thrust time: " << kernel_time << "ms\n";
 }

__global__ void cuda_blockwise_exclusive_sum(const uint32_t* g_idata, uint32_t* g_odata, uint32_t* block_sums, uint64_t n) {
    extern __shared__ uint32_t temp[];
    int thid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + thid;

    temp[thid] = (index > 0 && index < n) ? g_idata[index - 1] : 0;
    __syncthreads();

    for (int d = 1; d < blockDim.x; d *= 2) {
        uint32_t val = (thid >= d) ? temp[thid - d] : 0;
        __syncthreads();
        temp[thid] += val;
        __syncthreads();
    }

    if (index < n) {
        g_odata[index] = temp[thid];
    }

    if (thid == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = temp[thid];
    }
}

__global__ void cuda_adjust_block_sums(uint32_t* g_odata, uint32_t* block_sums, uint64_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t add_value = 0;

    for (int i = 0; i < blockIdx.x; ++i) {
        add_value += block_sums[i];
    }

    if (index < n) {
        g_odata[index] += add_value;
    }
}

__host__ void cuda_sum(const uint32_t* h_input, uint32_t* h_output, uint64_t N) {
    uint32_t *d_input, *d_output, *d_block_sums;
    size_t size = N * sizeof(uint32_t);

    const int block = 256;
    const int grid = (N + block - 1) / block;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Allocate memory on the device
    cudaCheckError(cudaMalloc(&d_input, size));
    cudaCheckError(cudaMalloc(&d_output, size));
    cudaCheckError(cudaMalloc(&d_block_sums, grid * sizeof(uint32_t)));

    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    size_t sharedMemSize = block * sizeof(uint32_t);
    cuda_blockwise_exclusive_sum<<<grid, block, sharedMemSize>>>(d_input, d_output, d_block_sums, N);
    cudaCheckError(cudaGetLastError());

    cuda_adjust_block_sums<<<grid, block>>>(d_output, d_block_sums, N);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(end);
    cudaCheckError(cudaDeviceSynchronize());
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel time: " << kernel_time << "ms\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
}


 __host__ void check_result(const uint32_t* w_ref, const uint32_t* w_opt,
                            const uint64_t size) {
   for (uint64_t i = 0; i < size; i++) {
     if (w_ref[i] != w_opt[i]) {
       cout << "Differences found between the two arrays.\n";
       assert(false);
     }
   }
   cout << "No differences found between base and test versions\n";
 }

 int main() {
   auto* h_input = new uint32_t[N];
   std::fill_n(h_input, N, 1);

   // Allocate memory for Thrust reference and CUDA output
   auto* h_thrust_ref = new uint32_t[N];
   auto* h_cuda_output = new uint32_t[N];
   std::fill_n(h_thrust_ref, N, 0);
   std::fill_n(h_cuda_output, N, 0);


   // Use Thrust code as reference

   thrust_sum(h_input, h_thrust_ref);

   // Use a CUDA kernel, time the execution
   cuda_sum(h_input, h_cuda_output, N);
  
   check_result(h_thrust_ref, h_cuda_output, N);

   delete[] h_thrust_ref;
   delete[] h_cuda_output;
   delete[] h_input;

   return EXIT_SUCCESS;
 }




