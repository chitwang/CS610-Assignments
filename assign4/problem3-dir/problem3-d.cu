#include <bits/stdc++.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/gather.h>
#include <cuda.h>
using namespace std;

#define NSEC_SEC_MUL (1.0e9)
#define ITER_CHUNK_SIZE (1 << 27)
#define NUM_VAR 10
#define THRESHOLD (std::numeric_limits<double>::epsilon())

struct ConstraintChecker {
    const double* constraints;
    const long long* loop_iter;
    const double* a;
    const double* b;

    __host__ __device__
    ConstraintChecker(const double* constraints, const long long* loop_iter, const double* a, const double* b)
        : constraints(constraints), loop_iter(loop_iter), a(a), b(b) {}

    __device__
    bool operator()(long long iter) const {
        long long iter_no[10];
        double x[10];
        double q[10] = {0.0};

        long long tmp_iter = iter;
        for (int i = 9; i >= 0; i--) {
            iter_no[i] = tmp_iter % loop_iter[i];
            tmp_iter /= loop_iter[i];
            x[i] = b[3 * i] + iter_no[i] * b[3 * i + 2];
        }

        bool flag = true;
        for (int i = 0; i < 10; i++) {
            q[i] = 0.0;
            for (int j = 0; j < 10; j++) {
                q[i] += a[i * 12 + j] * x[j];
            }
            q[i] -= a[i * 12 + 10];
            flag &= (fabs(q[i]) < constraints[i]);
        }
        return flag;
    }
};

int main() {
    thrust::host_vector<double> h_a(120);
    thrust::host_vector<double> h_b(30);

    FILE* fp = fopen("./disp.txt", "r");
    if (fp == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (int i = 0; i < 120 && fscanf(fp, "%lf", &h_a[i]) == 1; i++);
    fclose(fp);

    FILE* fpq = fopen("./grid.txt", "r");
    if (fpq == NULL) {
        printf("Error: could not open file\n");
        return 1;
    }
    for (int j = 0; j < 30 && fscanf(fpq, "%lf", &h_b[j]) == 1; j++);
    fclose(fpq);

    thrust::device_vector<double> d_a = h_a;
    thrust::device_vector<double> d_b = h_b;
    thrust::device_vector<double> d_constraints(10);
    thrust::device_vector<long long> d_loop_iter(11);

    double kk = 0.3;
    for (int i = 0; i < 10; i++) {
        d_constraints[i] = kk * d_a[11 + i * 12];
    }

    long long total_iter = 1;
    for (int i = 0; i < 10; i++) {
        d_loop_iter[i] = floor((d_b[3 * i + 1] - d_b[3 * i]) / d_b[3 * i + 2]);
        total_iter *= d_loop_iter[i];
    }
    d_loop_iter[10] = total_iter;
    long long result_cnt = 0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    std::ofstream output_file("results-vd.txt");
    output_file << std::fixed << std::setprecision(6);

    ConstraintChecker checker(thrust::raw_pointer_cast(d_constraints.data()),
                              thrust::raw_pointer_cast(d_loop_iter.data()),
                              thrust::raw_pointer_cast(d_a.data()),
                              thrust::raw_pointer_cast(d_b.data()));

    for (long long chunk_start = 0; chunk_start < total_iter; chunk_start += ITER_CHUNK_SIZE) {
        long long chunk_end = min(chunk_start + ITER_CHUNK_SIZE, total_iter);
        thrust::counting_iterator<long long> iter_begin(chunk_start);
        thrust::counting_iterator<long long> iter_end(chunk_end);

        thrust::device_vector<long long> valid_indices(chunk_end - chunk_start);
        auto valid_end = thrust::copy_if(
            iter_begin, iter_end, valid_indices.begin(), checker);

        int valid_points = valid_end - valid_indices.begin();
        result_cnt += valid_points;
        if (valid_points > 0) {
            thrust::host_vector<long long> h_valid_indices(valid_points);
            thrust::copy(valid_indices.begin(), valid_end, h_valid_indices.begin());

            for (int k = 0; k < valid_points; k++) {
                double x_array[NUM_VAR];
                long long tmp_iter = h_valid_indices[k];

                for (int i = NUM_VAR - 1; i >= 0; i--) {
                    x_array[i] = h_b[3 * i] + (tmp_iter % d_loop_iter[i]) * h_b[3 * i + 2];
                    tmp_iter /= d_loop_iter[i];
                }

                for (int l = 0; l < NUM_VAR; l++) {
                    output_file << x_array[l];
                    if (l < NUM_VAR - 1)
                        output_file << "\t";
                    else
                        output_file << "\n";
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
    cout << "Result pnts " << result_cnt << endl;
    return EXIT_SUCCESS;
}
