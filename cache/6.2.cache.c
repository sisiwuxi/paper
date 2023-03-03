#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <time.h> 

#define N 1000
double res1[N][N] __attribute__ ((aligned (64)));
double res2[N][N] __attribute__ ((aligned (64)));
double res3[N][N] __attribute__ ((aligned (64)));
double res4[N][N] __attribute__ ((aligned (64)));
double mul1[N][N] __attribute__ ((aligned (64)));
double mul2[N][N] __attribute__ ((aligned (64)));
#define CLS (64)
#define SM (CLS / sizeof (double))

void matrix_multiplication_1() {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res1[i][j] += mul1[i][k] * mul2[k][j];
      }
    }
  }
  return;
}

void matrix_multiplication_2() {
  double mul2T[N][N] __attribute__ ((aligned (64)));
  for (int j = 0; j < N; j++) {
    for (int k = 0; k < N; k++) {
      mul2T[j][k] = mul2[k][j];
    }
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res2[i][j] += mul1[i][k] * mul2T[j][k];
      }
    }
  }
  return;
}

void matrix_multiplication_3() {
  int i, i2, j, j2, k, k2;
  double *restrict rres;
  double *restrict rmul1;
  double *restrict rmul2;  
  for (i = 0; i < N; i += SM) {
    for (j = 0; j < N; j += SM) {
      for (k = 0; k < N; k += SM) {
        for (i2 = 0, rres = &res3[i][j], rmul1 = &mul1[i][k]; i2 < SM; ++i2, rres += N, rmul1 += N) {
          for (k2 = 0, rmul2 = &mul2[k][j]; k2 < SM; ++k2, rmul2 += N) {
            for (j2 = 0; j2 < SM; ++j2) {
              rres[j2] += rmul1[k2] * rmul2[j2];
            }
          }
        }
      }
    }
  }
  return;
}

void matrix_multiplication_4() {
  int i, i2, j, j2, k, k2;
  double *restrict rres;
  double *restrict rmul1;
  double *restrict rmul2;
  for (i = 0; i < N; i += SM) {
    for (j = 0; j < N; j += SM) {
      for (k = 0; k < N; k += SM) {
        for (i2 = 0, rres = &res4[i][j], rmul1 = &mul1[i][k]; i2 < SM; ++i2, rres += N, rmul1 += N) {
          _mm_prefetch (&rmul1[8], _MM_HINT_NTA);
          for (k2 = 0, rmul2 = &mul2[k][j]; k2 < SM; ++k2, rmul2 += N) {
            __m128d m1d = _mm_load_sd (&rmul1[k2]);
            m1d = _mm_unpacklo_pd (m1d, m1d);
            for (j2 = 0; j2 < SM; j2 += 2) {
              __m128d m2 = _mm_load_pd (&rmul2[j2]);
              __m128d r2 = _mm_load_pd (&rres[j2]);
              _mm_store_pd (&rres[j2],
              _mm_add_pd (_mm_mul_pd (m2, m1d), r2));
            }
          }
        }
      }
    }
  }
  return;
}

int main (void)
{
  // ... Initialize mul1 and mul2
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mul1[i][j] = rand();
      mul2[i][j] = rand();
      res1[i][j]  = 0.0f;
      res2[i][j]  = 0.0f;
      res3[i][j]  = 0.0f;
      res4[i][j]  = 0.0f;
    }
  }
  // mm1 Execution time: 6.074 seconds.
  // mm2 Execution time: 4.127 seconds.
  // mm3 Execution time: 3.754 seconds.
  // mm4 Execution time: 4.847 seconds.

  clock_t start_time = clock();
  matrix_multiplication_1();
  clock_t end_time = clock();
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
	printf("mm1 Execution time: %.3f seconds.\n", total_time); 

  start_time = clock();
  matrix_multiplication_2();
  end_time = clock();
  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
  printf("mm2 Execution time: %.3f seconds.\n", total_time); 

  start_time = clock();
  matrix_multiplication_3();
  end_time = clock();
  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
  printf("mm3 Execution time: %.3f seconds.\n", total_time); 

  start_time = clock();
  matrix_multiplication_4();
  end_time = clock();
  total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
  printf("mm4 Execution time: %.3f seconds.\n", total_time); 

  // ... use res matrix
  int sum_1_2 = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum_1_2 += res1[i][j] - res2[i][j];
    }
  }
  int sum_1_3 = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum_1_3 += res1[i][j] - res3[i][j];
    }
  }  
  int sum_1_4 = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum_1_4 += res1[i][j] - res4[i][j];
    }
  }
  printf("\nsum=%d, %d, %d\n", sum_1_2, sum_1_3, sum_1_4);
  return 0;
}
