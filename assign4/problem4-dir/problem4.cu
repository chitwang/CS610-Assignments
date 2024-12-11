#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <numeric>
#include <sys/time.h>
#include <cmath>
#include <assert.h>

#define THRESHOLD (std::numeric_limits<float>::epsilon())
#define N 64  // size of each dimension
#define FILTER_SIZE 3

using std::cerr;
using std::cout;
using std::endl;
const uint64_t TILE = 8;

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CPU version of 2D convolution
void cpu_convolution_2D(const float* input, float* output, int width) {
    int half_filter = FILTER_SIZE / 2;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            float sum = 0.0;
            int count = FILTER_SIZE * FILTER_SIZE;

            for (int i = -half_filter; i <= half_filter; ++i) {
                for (int j = -half_filter; j <= half_filter; ++j) {
                    int nx = x + i;
                    int ny = y + j;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
                        sum += input[nx * width + ny];
                    }
                }
            }
            output[x * width + y] = sum / count;
        }
    }
}

void cpu_convolution_3D(const float* input, float* output, int width) {
    int half_filter = FILTER_SIZE / 2;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < width; y++) {
            for (int z = 0; z < width; z++) {
                float sum = 0.0;
                int count = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

                for (int i = -half_filter; i <= half_filter; ++i) {
                    for (int j = -half_filter; j <= half_filter; ++j) {
                        for (int k = -half_filter; k <= half_filter; ++k) {
                            int nx = x + i;
                            int ny = y + j;
                            int nz = z + k;
                            if (nx >= 0 && nx < width && ny >= 0 && ny < width && nz >= 0 && nz < width) {
                                sum += input[nx * width * width + ny * width + nz];
                            }
                        }
                    }
                }
                output[x * width * width + y * width + z] = sum / count;
            }
        }
    }
}


__global__ void basic_convolution_2D(const float* input, float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half_filter = FILTER_SIZE / 2;
    float sum = 0.0;
    int count = FILTER_SIZE * FILTER_SIZE;

    for (int i = -half_filter; i <= half_filter; ++i) {
        for (int j = -half_filter; j <= half_filter; ++j) {
            int nx = x + i;
            int ny = y + j;
            if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
                sum += input[nx * width + ny];
            }
        }
    }
    if (x < width && y < width) {
        output[x * width + y] = sum / count;
    }
}

__global__ void optimized_convolution_2D(const float* input, float* output, int width) {
    __shared__ double tile[TILE + FILTER_SIZE][TILE + FILTER_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int half_filter = FILTER_SIZE/2;

    int x = blockIdx.x * TILE + tx - half_filter;
    int y = blockIdx.y * TILE + ty - half_filter;

    if (x >= 0 && x < width && y >= 0 && y < width) {
        tile[tx][ty] = input[x * width + y];
    } else {
        tile[tx][ty] = 0.0f; 
    }

    __syncthreads();
    int count = FILTER_SIZE * FILTER_SIZE;

    if(tx >= half_filter and tx < TILE + half_filter and ty >= half_filter and ty < TILE + half_filter){
        float sum = 0;
        for(int i= -half_filter; i<=half_filter; i++){
            for(int j=-half_filter; j<=half_filter; j++){
                sum += tile[tx+i][ty+j];
            }
        }
        output[x*width + y] = sum/count;
    }
}

__global__ void basic_convolution_3D(const float* input, float* output, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    int half_filter = FILTER_SIZE / 2;
    float sum = 0.0;
    int count = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

    for (int i = -half_filter; i <= half_filter; ++i) {
        for (int j = -half_filter; j <= half_filter; ++j) {
            for (int k = -half_filter; k <= half_filter; ++k) {
                int nx = x + i;
                int ny = y + j;
                int nz = z + k;
                if (nx >= 0 && nx < width && ny >= 0 && ny < width && nz >= 0 && nz < width) {
                    sum += input[(nx * width * width) + (ny * width) + nz];
                }
            }
        }
    }
    if (x < width && y < width && z < width) {
        output[(x * width * width) + (y * width) + z] = sum / count;
    }
}

__global__ void optimized_convolution_3D(const float* input, float* output, int width) {
    __shared__ double tile[(TILE + FILTER_SIZE)*(TILE + FILTER_SIZE)*(TILE + FILTER_SIZE)];
    int shared_width = TILE + FILTER_SIZE;
    int shared_area = shared_width * shared_width;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int half_filter = FILTER_SIZE/2;

    int x = blockIdx.x * TILE + tx - half_filter;
    int y = blockIdx.y * TILE + ty - half_filter;
    int z = blockIdx.z * TILE + tz - half_filter;

    if (x >= 0 && x < width && y >= 0 && y < width && z >= 0 && z < width) {
        tile[tx * shared_area + ty * shared_width + tz] = input[x * width * width + y * width + z];
    } else {
        tile[tx * shared_area + ty * shared_width + tz] = 0.0f; 
    }

    __syncthreads();

    int count = FILTER_SIZE * FILTER_SIZE * FILTER_SIZE;

    if(tx >= half_filter and tx < TILE + half_filter and ty >= half_filter and ty < TILE + half_filter and tz >= half_filter and tz < TILE + half_filter){
        float sum = 0;
        #pragma unroll
        for(int i= -half_filter; i<=half_filter; i++){
            for(int j=-half_filter; j<=half_filter; j++){
                for(int k=-half_filter; k<=half_filter; k++){
                    sum += tile[(tx+i)*shared_area + (ty+j)*shared_width + tz+k];
                }
            }
        }
        output[x*width*width + y*width + z] = sum/count;
    }
}



void check_result(const float* ref, const float* opt, int size) {
    int numdiffs = 0;
    float maxdiff = 0.0f;

    for (int i = 0; i < size; i++) {
        float this_diff = std::fabs(ref[i] - opt[i]);
        if (this_diff > THRESHOLD) {
            numdiffs++;
            if (this_diff > maxdiff) {
                maxdiff = this_diff;
            }
        }
    }

    if (numdiffs > 0) {
        cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff << endl;
    } else {
        cout << "No differences found between CPU and GPU results\n";
    }
}

void print2D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << "\t";
    }
    cout << "n";
  }
}

void print3D(const float* A) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "n";
    }
    cout << "n";
  }
}

double rtclock() {
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) {
        cout << "Error return from gettimeofday: " << stat << "\n";
    }
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

int driver(int dim) {
    int size = pow(N, dim);
    float* h_input = new float[size];
    float* h_output_cpu = new float[size]();  // Initialize to zero
    float* h_output_gpu = new float[size]();  // Initialize to zero

    // Initialize input array
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Run CPU version
    double clkbegin = rtclock();
    if (dim == 2)
        cpu_convolution_2D(h_input, h_output_cpu, N);
    else if (dim == 3)
        cpu_convolution_3D(h_input, h_output_cpu, N);
    double clkend = rtclock();
    cout << "CPU convolution time: " << (clkend - clkbegin) * 1000 << " ms\n";

    // Allocate device memory
    float *d_input, *d_output;
    cudaCheckError(cudaMalloc((void**)&d_input, size * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_output, size * sizeof(float)));

    // Kernel configurations
    dim3 block2d(8,8);
    dim3 grid2d((N + block2d.x - 1) / block2d.x, (N + block2d.y - 1) / block2d.y);

    dim3 block3d(8, 8, 8);
    dim3 grid3d((N + block3d.x - 1) / block3d.x, (N + block3d.y - 1) / block3d.y, (N + block3d.z - 1) / block3d.z);

    dim3 block2do(TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1);
    dim3 grid2do((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    dim3 block3do(TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1, TILE+FILTER_SIZE-1);
    dim3 grid3do((N + TILE - 1) / TILE, (N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, end;
    cudaEvent_t start_kernel, end_kernel;
    float kernel_time, overall_time;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    
    // Run basic convolution kernel
    cudaEventRecord(start);
    // Copy input data to device
    cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    if (dim == 2) {
        cudaEventRecord(start_kernel);
        basic_convolution_2D<<<grid2d, block2d>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    } else if (dim == 3) {
        cudaEventRecord(start_kernel);
        basic_convolution_3D<<<grid3d, block3d>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    }
    cudaCheckError(cudaGetLastError());  // Check for launch errors
    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&overall_time, start, end);
    cout << "Basic Kernel time: " << kernel_time << "ms\n";
    cout << "Basic Kernel time including memory transfers: " << overall_time << "ms\n";
    // Copy output data back to host and check results
    check_result(h_output_cpu, h_output_gpu, size);

    cudaEventDestroy(start_kernel);
    cudaEventDestroy(start);
    cudaEventDestroy(end_kernel);
    cudaEventDestroy(end);

    // Run optimized convolution kernel
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&end_kernel);
    cudaEventRecord(start);
    cudaCheckError(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    if (dim == 2) {
        cudaEventRecord(start_kernel);
        optimized_convolution_2D<<<grid2do, block2do>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    } else if (dim == 3) {
        cudaEventRecord(start_kernel);
        optimized_convolution_3D<<<grid3do, block3do>>>(d_input, d_output, N);
        cudaEventRecord(end_kernel);
        cudaCheckError(cudaEventSynchronize(end_kernel));
        cudaEventElapsedTime(&kernel_time, start_kernel, end_kernel);
    }
    cudaCheckError(cudaGetLastError());  // Check for launch errors

    cudaCheckError(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&overall_time, start, end);
    cout << "Optimized Kernel time: " << kernel_time << "ms\n";
    cout << "Optimized Kernel time including memory transfers: " << overall_time << "ms\n";

    // Copy output data back to host and verify results
    check_result(h_output_cpu, h_output_gpu, size);

    // Free memory   
    cudaFree(d_output);
    cudaFree(d_input);
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    return EXIT_SUCCESS;
}


int main() {
    cout << "######### 2D Convolution ##########\n";
    assert(driver(2) == EXIT_SUCCESS);
    cout << "######### 3D Convolution ##########\n";
    assert(driver(3) == EXIT_SUCCESS);

    return EXIT_SUCCESS;
}
