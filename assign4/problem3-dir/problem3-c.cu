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
            #pragma unroll
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
    double *a, *b;
    int i, j;
    cudaCheckError(cudaMallocManaged(&a, 120*sizeof(double)));
    cudaCheckError(cudaMallocManaged(&b, 30*sizeof(double)));

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
    double *constraints;
    cudaCheckError(cudaMallocManaged(&constraints, NUM_VAR * sizeof(double)));
    for (i = 0; i < NUM_VAR; i++) {
        constraints[i] = kk * a[11 + i * 12];
    }

    long long *loop_iter;
    cudaCheckError(cudaMallocManaged(&loop_iter, (NUM_VAR+1)*sizeof(double)));
    long long total_iter = 1;
    for (i = 0; i < NUM_VAR; i++) {
        loop_iter[i] = floor((b[3 * i + 1] - b[3 * i]) / b[3 * i + 2]);
        total_iter *= loop_iter[i];
    }
    loop_iter[NUM_VAR] = total_iter;
    long long result_cnt = 0;
    int *output_count;
    long long *output_x;
    cudaMallocManaged(&output_count, sizeof(int));
    cudaMallocManaged(&output_x, ITER_CHUNK_SIZE * sizeof(long long));

    int block = 512;
    int grid = (1 << 16);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    ofstream output_file("results-vc.txt");
    output_file << setprecision(6) << fixed;

    for (long long chunk_start = 0; chunk_start < total_iter; chunk_start += ITER_CHUNK_SIZE) {
        long long chunk_end = min(chunk_start + ITER_CHUNK_SIZE, total_iter);

        *output_count = 0;
        kernel<<<grid, block>>>(constraints, loop_iter, a, b, output_x, output_count, chunk_start, chunk_end);
        cudaDeviceSynchronize();
        
        result_cnt += (*output_count);
        if (*output_count > 0) {
            sort(output_x, output_x + (*output_count));
            for (int k = 0; k < (*output_count); k ++) {
                double x_array[NUM_VAR];
                long long tmp_iter = output_x[k];
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
  
    cudaFree(a);
    cudaFree(b);
    cudaFree(constraints);
    cudaFree(loop_iter);
    cudaFree(output_count);
    cudaFree(output_x);
    return EXIT_SUCCESS;
}