#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <immintrin.h>
#include <x86intrin.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 30)
#define SSE_WIDTH_BITS (128)
#define AVX2_WIDTH_BITS (256)
#define ALIGN (32)

/** Helper methods for debugging */

void print_array(const int* array) {
  for (int i = 0; i < N; i++) {
    cout << array[i] << "\t";
  }
  cout << "\n";
}

void print128i_u32(__m128i var, int start) {
  alignas(ALIGN) uint32_t val[4];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [" << start << ":" << start + 3 << "]: " << val[0] << " "
       << val[1] << " " << val[2] << " " << val[3] << "\n";
}

void print128i_u64(__m128i var) {
  alignas(ALIGN) uint64_t val[2];
  _mm_store_si128((__m128i*)val, var);
  cout << "Values [0:1]: " << val[0] << " " << val[1] << "\n";
}

void print_m256i(const __m256i& reg) {
    // Create an array to store the values
    alignas(ALIGN) int values[8];
    
    // Store the values from the AVX register into the array
    _mm256_store_si256((__m256i*)values, reg);
    
    // Print the values
    std::cout << "AVX2 register contents: ";
    for (int i = 0; i < 8; i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;
}

__attribute__((optimize("no-tree-vectorize"))) int
ref_version(int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
  for (int i = 0; i < N; i++) {
    tmp += source[i];
    dest[i] = tmp;
  }
  return tmp;
}

int omp_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  int tmp = 0;
#pragma omp simd reduction(inscan, + : tmp)
  for (int i = 0; i < N; i++) {
    tmp += source[i];
#pragma omp scan inclusive(tmp)
    dest[i] = tmp;
  }
  return tmp;
}

// Tree reduction idea on every 128 bits vector data, involves 2 shifts, 3 adds,
// one broadcast
int sse4_version(const int* __restrict__ source, int* __restrict__ dest) {
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  // Return vector of type __m128i with all elements set to zero, to be added as
  // previous sum for the first four elements.
  __m128i offset = _mm_setzero_si128();

  const int stride = SSE_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {
    // Load 128-bits of integer data from memory into x. source_addr must be
    // aligned on a 16-byte boundary to be safe.
    __m128i x = _mm_load_si128((__m128i*)&source[i]);
    // Let the numbers in x be [d,c,b,a], where a is at source[i].
    __m128i tmp0 = _mm_slli_si128(x, 4);
    // Shift x left by 4 bytes while shifting in zeros. tmp0 becomes [c,b,a,0].
    __m128i tmp1 =
        _mm_add_epi32(x, tmp0); // Add packed 32-bit integers in x and tmp0.
    // tmp1 becomes [c+d,b+c,a+b,a].
    // Shift tmp1 left by 8 bytes while shifting in zeros.
    __m128i tmp2 = _mm_slli_si128(tmp1, 8); // tmp2 becomes [a+b,a,0,0].
    // Add packed 32-bit integers in tmp2 and tmp1.
    __m128i out = _mm_add_epi32(tmp2, tmp1);
    // out contains [a+b+c+d,a+b+c,a+b,a].
    out = _mm_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by
    // offset.
    // Store 128-bits of integer data from out into memory. dest_addr must be
    // aligned on a 16-byte boundary to be safe.
    _mm_store_si128((__m128i*)&dest[i], out);
    // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the
    // formula (z << 6) | (y << 4) | (x << 2) | w.
    int mask = _MM_SHUFFLE(3, 3, 3, 3);
    // Bits [7:0] of mask are 11111111 to pick the third integer (11) from out
    // (i.e., a+b+c+d).

    // Shuffle 32-bit integers in out using the control in mask.
    offset = _mm_shuffle_epi32(out, mask);
    // offset now contains 4 copies of a+b+c+d.
  }
  return dest[N - 1];
}


/// num is the number of integers by which shifting should be done 
__m256i shift_left(__m256i r, int num){
  for(int i=0;i<num;i++){
    __m256i r1 = _mm256_permute2f128_si256(r, r, 0x08);
    r =  _mm256_alignr_epi8(r, r1, 12);
  }
  return r;
}


int avx2_version(const int* __restrict__ source, int* __restrict__ dest) { 
  __builtin_assume_aligned(source, ALIGN);
  __builtin_assume_aligned(dest, ALIGN);

  __m256i offset = _mm256_setzero_si256();

  const int stride = AVX2_WIDTH_BITS / (sizeof(int) * CHAR_BIT);
  for (int i = 0; i < N; i += stride) {

    __m256i x = _mm256_load_si256((__m256i*)&source[i]);
    // Let the numbers in x be [h,g,f,e,d,c,b,a], where a is at source[i].
    __m256i tmp0 = shift_left(x, 1);
    // print_m256i(tmp0);
    //tmp0 = [g,f,e,d,c,b,a,0]

    __m256i tmp1 = _mm256_add_epi32(x, tmp0); // Add packed 32-bit integers in x and tmp0.
    // tmp1 becomes [g+h, f+g, e+f, d+e, c+d,b+c,a+b,a].

    // Shift tmp1 left by 8 bytes while shifting in zeros.
    tmp0 = shift_left(tmp1, 2);
    __m256i tmp2 =  _mm256_add_epi32(tmp1, tmp0); // tmp2 becomes [e+f+g+h, d+e+f+g, e+f+c+d, d+e+b+c, c+d+a+b, b+c+a,a+b,a].
    
    tmp0 = shift_left(tmp2, 4);
    __m256i out = _mm256_add_epi32(tmp0, tmp2);  // [a..h, a..g, a..f, a..e, a..d, a..c, a..b, a]
  
    out = _mm256_add_epi32(out, offset);
    // out now includes the sum from the previous set of numbers, given by
    // offset.
    // Store 256-bits of integer data from out into memory. dest_addr must be
    // aligned on a 32-byte boundary to be safe.
    _mm256_store_si256((__m256i*)&dest[i], out);
    // _MM_SHUFFLE(z, y, x, w) macro forms an integer mask according to the
    // formula (z << 6) | (y << 4) | (x << 2) | w.
    int mask = _MM_SHUFFLE(3, 3, 3, 3);
    
    __m128i high_offset = _mm256_extracti128_si256(out, 1);   // 4 integers bigger prefix sums [a..g, a..f, a..e, a..d]
    __m128i largest_sum = _mm_shuffle_epi32(high_offset, mask);

    offset = _mm256_castsi128_si256(largest_sum);
    offset = _mm256_inserti128_si256(offset, largest_sum, 1);
  }
  return dest[N - 1];
 }

__attribute__((optimize("no-tree-vectorize"))) int main() {
  int* array = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(array, array + N, 1);

  int* ref_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(ref_res, ref_res + N, 0);
  HRTimer start = HR::now();
  int val_ser = ref_version(array, ref_res);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial version: " << val_ser << " time: " << duration << endl;

  int* omp_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(omp_res, omp_res + N, 0);
  start = HR::now();
  int val_omp = omp_version(array, omp_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_omp || printf("OMP result is wrong!\n"));
  cout << "OMP version: " << val_omp << " time: " << duration << endl;
  free(omp_res);

  int* sse_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(sse_res, sse_res + N, 0);
  start = HR::now();
  int val_sse = sse4_version(array, sse_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_sse || printf("SSE result is wrong!\n"));
  cout << "SSE version: " << val_sse << " time: " << duration << endl;
  free(sse_res);

  int *avx2_res = static_cast<int*>(aligned_alloc(ALIGN, N * sizeof(int)));
  std::fill(avx2_res, avx2_res + N, 0);
  start = HR::now();
  int val_avx2 = avx2_version(array, avx2_res);
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  assert(val_ser == val_avx2 || printf("AVX2 result is wrong!\n"));
  cout << "AVX2 version: " << val_avx2 << " time: " << duration << endl;

  return EXIT_SUCCESS;
}
