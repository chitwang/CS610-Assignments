#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

#define NSEC_SEC_MUL (1.0e9)
#define ITER_CHUNK_SIZE (1 << 25)
#define NUM_VAR 10
#define THRESHOLD (std::numeric_limits<double>::epsilon())

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel to compute valid results and store x arrays in ordered buffer
__global__ void kernel(double *constraints, long long *loop_iter, double *dev_a, double *dev_b, long long *dev_output_x, int *dev_output_count, long long chunk_start, long long chunk_end) {
    int tx = threadIdx.x;
    int x = blockIdx.x * blockDim.x + tx;  // Global thread ID
    long long total_threads = gridDim.x * blockDim.x;
    long long iter_per_thread = (chunk_end - chunk_start + total_threads - 1) / total_threads;
    long long start_iter = chunk_start + iter_per_thread * x;
    long long end_iter = min(chunk_end - 1, start_iter + iter_per_thread - 1);

    double x_array[NUM_VAR];
    double q[NUM_VAR] = {0.0};
    long long iter_no[NUM_VAR];

    for (long long iter = start_iter; iter <= end_iter; iter++) {
        long long tmp_iter = iter;

        for (int i = NUM_VAR - 1; i >= 0; i--) {
            iter_no[i] = tmp_iter % loop_iter[i];
            tmp_iter /= loop_iter[i];
            x_array[i] = dev_b[3 * i] + iter_no[i] * dev_b[3 * i + 2];
        }

        bool is_valid = true;
        for (int i = 0; i < NUM_VAR; i++) {
            q[i] = 0.0;
            for (int j = 0; j < NUM_VAR; j++) {
                q[i] += dev_a[i * 12 + j] * x_array[j];
            }
            q[i] -= dev_a[i * 12 + 10];
            is_valid &= (fabs(q[i]) <= constraints[i]);
        }
        if (is_valid) {
          int old = atomicAdd(dev_output_count, 1); 
          dev_output_x[old] = iter;
        }
    }
}

int main() {
    double a[120], b[30];
    int i, j;

    FILE* fp = fopen("./disp.txt", "r");
    if (fp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (i = 0; !feof(fp) && i < 120; i++) {
        if (!fscanf(fp, "%lf", &a[i])) {
            printf("Error reading disp.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);

    FILE* fpq = fopen("./grid.txt", "r");
    if (fpq == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (j = 0; !feof(fpq) && j < 30; j++) {
        if (!fscanf(fpq, "%lf", &b[j])) {
            printf("Error reading grid.txt\n");
            exit(EXIT_FAILURE);
        }
    }
    fclose(fpq);

    double kk = 0.3;
    double constraints[NUM_VAR];
    for (i = 0; i < NUM_VAR; i++) {
        constraints[i] = kk * a[11 + i * 12];
    }

    long long loop_iter[NUM_VAR+1];
    long long total_iter = 1;
    for (i = 0; i < NUM_VAR; i++) {
        loop_iter[i] = floor((b[3 * i + 1] - b[3 * i]) / b[3 * i + 2]);
        total_iter *= loop_iter[i];
    }
    loop_iter[NUM_VAR] = total_iter;
    long long result_cnt = 0;
    long long *host_output_x = new long long[ITER_CHUNK_SIZE]();

    // Allocate device memory
    double *dev_a, *dev_b, *dev_constraints;
    long long *dev_loop_iter;
    int *dev_output_count;
    long long *dev_output_x;

    cudaMalloc(&dev_a, 120 * sizeof(double));
    cudaMalloc(&dev_b, 30 * sizeof(double));
    cudaMalloc(&dev_constraints, NUM_VAR * sizeof(double));
    cudaMalloc(&dev_loop_iter, (NUM_VAR+1) * sizeof(long long));
    cudaMalloc(&dev_output_count, sizeof(int));
    cudaMalloc(&dev_output_x, ITER_CHUNK_SIZE * sizeof(long long));

    cudaMemcpy(dev_a, a, 120 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, 30 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_constraints, constraints, NUM_VAR * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_loop_iter, loop_iter, (NUM_VAR+1) * sizeof(long long), cudaMemcpyHostToDevice);

    int block = 512;
    int grid = (1 << 16);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    ofstream output_file("results-va.txt");
    output_file << setprecision(6) << fixed;

    for (long long chunk_start = 0; chunk_start < total_iter; chunk_start += ITER_CHUNK_SIZE) {
        long long chunk_end = min(chunk_start + ITER_CHUNK_SIZE, total_iter);

        cudaMemset(dev_output_count, 0, sizeof(int));
        kernel<<<grid, block>>>(dev_constraints, dev_loop_iter, dev_a, dev_b, dev_output_x, dev_output_count, chunk_start, chunk_end);
        cudaDeviceSynchronize();

        int output_count;
        cudaMemcpy(&output_count, dev_output_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        result_cnt += output_count;
        if (output_count > 0) {
            cudaMemcpy(host_output_x, dev_output_x, output_count * sizeof(long long), cudaMemcpyDeviceToHost);
            sort(host_output_x, host_output_x + output_count);
            for (int k = 0; k < output_count; k ++) {
                double x_array[NUM_VAR];
                long long tmp_iter = host_output_x[k];
                for(int i=NUM_VAR-1; i>=0; i--){
                    x_array[i] = b[3 * i] + (tmp_iter % loop_iter[i]) * b[3 * i + 2];
                    tmp_iter /= loop_iter[i];
                }
                for (int l = 0; l < NUM_VAR; l++) {
                    output_file << x_array[l];
                    if(l == NUM_VAR-1)
                      output_file << std::endl;
                    else 
                      output_file << "\t";
                }
            }
        }
    }

    output_file.close();

    cudaEventRecord(end);
    cudaDeviceSynchronize();
    float kernel_time = 0.0;
    cudaEventElapsedTime(&kernel_time, start, end);
    std::cout << "Kernel time " << kernel_time * 1e-3 << "s\n";
    std::cout << "Result pnts " << result_cnt << std::endl;
  
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_constraints);
    cudaFree(dev_loop_iter);
    cudaFree(dev_output_count);
    cudaFree(dev_output_x);
    delete [] host_output_x;
    return EXIT_SUCCESS;
}