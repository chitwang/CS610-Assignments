#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <immintrin.h>
#include <string>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

const static float EPSILON = std::numeric_limits<float>::epsilon();
bool IS_ALIGNED_VARIANT = true;

#define N (1024)
#define ALIGN (32)

void matmul_seq(float **A, float **B, float **C)
{
  float sum = 0;
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      sum = 0;
      for (int k = 0; k < N; k++)
      {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}

void matmul_sse4(float**A, float **B, float **C)
{
  // ikj variant of matrix multiplication
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      __m128 a_ik = _mm_set_ps1(A[i][k]);
      for (int j = 0; j < N; j += 4)
      {
        __m128 b_kj = _mm_loadu_ps(&B[k][j]);
        __m128 c_ij = _mm_loadu_ps(&C[i][j]);
        __m128 res = _mm_mul_ps(a_ik, b_kj);
        c_ij = _mm_add_ps(res, c_ij);
        _mm_storeu_ps(&C[i][j], c_ij);
      }
    }
  }
}

void matmul_sse4_aligned(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C)
{
  __builtin_assume_aligned(A, ALIGN);
  __builtin_assume_aligned(B, ALIGN);
  __builtin_assume_aligned(C, ALIGN);
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      __m128 a_ik = _mm_set_ps1(A[i][k]);
      for (int j = 0; j < N; j += 4)
      {
        __m128 b_kj = _mm_load_ps(&B[k][j]);
        __m128 c_ij = _mm_load_ps(&C[i][j]);
        __m128 res = _mm_mul_ps(a_ik, b_kj);
        c_ij = _mm_add_ps(res, c_ij);
        _mm_store_ps(&C[i][j], c_ij);
      }
    }
  }
}

void matmul_avx2(float **A, float **B, float **C)
{
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      __m256 a_ik = _mm256_broadcast_ss(&A[i][k]);
      for (int j = 0; j < N; j += 8)
      {
        __m256 b_kj = _mm256_loadu_ps(&B[k][j]);
        __m256 c_ij = _mm256_loadu_ps(&C[i][j]);
        __m256 res = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
        _mm256_storeu_ps(&C[i][j], res);
      }
    }
  }
}

void matmul_avx2_aligned(float** __restrict__ A, float** __restrict__ B, float** __restrict__ C)
{
   __builtin_assume_aligned(A, ALIGN);
   __builtin_assume_aligned(B, ALIGN);
   __builtin_assume_aligned(C, ALIGN);
  for (int i = 0; i < N; i++)
  {
    for (int k = 0; k < N; k++)
    {
      __m256 a_ik = _mm256_broadcast_ss(&A[i][k]);
      for (int j = 0; j < N; j += 8)
      {
        __m256 b_kj = _mm256_load_ps(&B[k][j]);
        __m256 c_ij = _mm256_load_ps(&C[i][j]);
        __m256 res = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
        _mm256_store_ps(&C[i][j], res);
      }
    }
  }
}

void check_result(float **w_ref, float **w_opt)
{
  float maxdiff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      float this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > EPSILON)
      {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0)
  {
    cout << numdiffs << " Diffs found over THRESHOLD " << EPSILON
         << "; Max Diff = " << maxdiff << endl;
  }
  else
  {
    cout << "No differences found between base and test versions\n";
  }
}

int main(int argc, char *argv[])
{
  float **A, **B, **C_seq, **C_sse4, **C_avx2;
  IS_ALIGNED_VARIANT = std::stoi(argv[1]);
  if (IS_ALIGNED_VARIANT)
  {
    A = static_cast<float **>(aligned_alloc(ALIGN, N*sizeof(float*)));
    B = static_cast<float **>(aligned_alloc(ALIGN, N*sizeof(float*)));
    C_seq = static_cast<float **>(aligned_alloc(ALIGN, N*sizeof(float*)));
    C_sse4 = static_cast<float **>(aligned_alloc(ALIGN, N*sizeof(float*)));
    C_avx2 = static_cast<float **>(aligned_alloc(ALIGN, N*sizeof(float*)));
    for (int i = 0; i < N; i++)
    {
      A[i] = static_cast<float *>(aligned_alloc(ALIGN, N*sizeof(float)));
      B[i] = static_cast<float *>(aligned_alloc(ALIGN, N*sizeof(float)));
      C_seq[i] = static_cast<float *>(aligned_alloc(ALIGN, N*sizeof(float)));
      C_sse4[i] = static_cast<float *>(aligned_alloc(ALIGN, N*sizeof(float)));
      C_avx2[i] = static_cast<float *>(aligned_alloc(ALIGN, N*sizeof(float)));
    }
  }
  else
  {
    A = new float *[N];
    for (int i = 0; i < N; i++)
    {
      A[i] = new float[N]();
    }
    B = new float *[N];
    for (int i = 0; i < N; i++)
    {
      B[i] = new float[N]();
    }

    C_seq = new float *[N];
    C_sse4 = new float *[N];
    C_avx2 = new float *[N];
    for (int i = 0; i < N; i++)
    {
      C_seq[i] = new float[N]();
      C_sse4[i] = new float[N]();
      C_avx2[i] = new float[N]();
    }
  }
  // initialize arrays
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      A[i][j] = 0.1F;
      B[i][j] = 0.2F;
      C_seq[i][j] = 0.0F;
      C_sse4[i][j] = 0.0F;
      C_avx2[i][j] = 0.0F;
    }
  }
  HRTimer start = HR::now();
  matmul_seq(A, B, C_seq);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul seq time: " << duration << " ms" << endl;

  start = HR::now();
  (IS_ALIGNED_VARIANT ? matmul_sse4_aligned(A, B, C_sse4) : matmul_sse4(A, B, C_sse4));
  end = HR::now();
  check_result(C_seq, C_sse4);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul SSE4 time: " << duration << " ms" << endl;

  start = HR::now();
  (IS_ALIGNED_VARIANT ? matmul_avx2_aligned(A, B, C_avx2) : matmul_avx2(A, B, C_avx2));
  end = HR::now();
  check_result(C_seq, C_avx2);
  duration = duration_cast<milliseconds>(end - start).count();
  cout << "Matmul AVX2 time: " << duration << " ms" << endl;

  return EXIT_SUCCESS;
}
