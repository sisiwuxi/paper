//
// https://utcc.utoronto.ca/~cks/space/blog/linux/KernelSegfaultErrorCodes
//
#include<mmintrin.h> //mmx, Multi Media eXtension
#include<xmmintrin.h> //sse, Streaming SIMD Extensions
#include<emmintrin.h> //sse2
#include<pmmintrin.h> //sse3
#include<stdio.h>
// AVX, Advanced Vector Extensions
#include <stdio.h> 
#include <stdlib.h> 
#include <sys/mman.h> 
#include <sys/resource.h> 
#include <time.h> 

// getconf -a | grep CACHE
# define ITERS (1000) // 10000000
// # define ARRAY_SIZE (1024 * 32)
// #define MAX_WS_SIZE (1024 * 1024 * 512) // 512 MB 
# define MAX_WS_SIZE (ARRAY_SIZE * 2)
# define ALIGNMENT (32) // >=sizeof(void*)=8

int test_sse(int ARRAY_SIZE) {  
	printf("\n=========================== test_sse ===========================\n");
	float *a = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 
	float *b = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 
	float *c = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 
	printf("a-aligned addr:   %p\n", (void*)a);
	printf("b-aligned addr:   %p\n", (void*)b);
	printf("c-aligned addr:   %p\n", (void*)c);
	// set Working Set Size
	struct rlimit rlim = {0}; 
	rlim.rlim_cur = MAX_WS_SIZE; 
	rlim.rlim_max = MAX_WS_SIZE; 
	if (setrlimit(RLIMIT_AS, &rlim) != 0) { 
		printf("Error: failed to set Working Set size.\n"); 
		return -1; 
	} 
	// initialize a & b
	srand(time(NULL)); 
	for (int i = 0; i < ARRAY_SIZE; i++) { 
		a[i] = rand(); 
		b[i] = rand();
		c[i] = 0.0f;
	} 
	
	clock_t start_time = clock();  
	__m128 *p1, *p2, *p3;  
	for(int j=0; j<ITERS; j++) {   
		p1 = (__m128 *)a;   
		p2 = (__m128 *)b;   
		p3 = (__m128 *)c;    
		for(int i=0; i<ARRAY_SIZE/4; i++) {
			// ps = p1*p2 + p3
			*p3 = _mm_add_ps(_mm_mul_ps(*p1, *p2), *p3);   
			p1++;
			p2++;
			p3++;   
		}  
	}
	clock_t end_time = clock(); 
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
	printf("Execution time: %.3f seconds.\n", total_time); 
	free(a); free(b); free(c);
	return 0;
}

int test2(int ARRAY_SIZE) {
	printf("\n=========================== test2 ===========================\n");
	// int* a = (int*)malloc(ARRAY_SIZE * sizeof(int)); 
	// int* b = (int*)malloc(ARRAY_SIZE * sizeof(int)); 
	// int* c = (int*)malloc(ARRAY_SIZE * sizeof(int));
	// int* a = (int*)aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(int)); 
	// int* b = (int*)aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(int)); 
	// int* c = (int*)aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(int)); 
	float *a = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 
	float *b = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 
	float *c = reinterpret_cast<float*>(aligned_alloc(ALIGNMENT, ARRAY_SIZE * sizeof(float))); 

	printf("a addr:   %p\n", (void*)a);
	printf("b addr:   %p\n", (void*)b);
	printf("c addr:   %p\n", (void*)c);	 
	// set Working Set Size
	struct rlimit rlim = {0}; 
	rlim.rlim_cur = MAX_WS_SIZE; 
	rlim.rlim_max = MAX_WS_SIZE; 
	if (setrlimit(RLIMIT_AS, &rlim) != 0) { 
		printf("Error: failed to set Working Set size.\n"); 
		return -1; 
	} 
	// initialize a & b 
	srand(time(NULL)); 
	for (int i = 0; i < ARRAY_SIZE; i++) { 
		a[i] = rand(); 
		b[i] = rand(); 
	} 
	clock_t start_time = clock(); 
	for(int j=0; j<ITERS; j++) {
		for (int i = 0; i < ARRAY_SIZE; i++) { 
			c[i] = a[i] * b[i] + c[i]; 
		} 
	}
	clock_t end_time = clock(); 
	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC; 
	printf("Execution time: %.3f seconds.\n", total_time); 
	free(a); free(b); free(c);
	return 0;
}

void test1() {
	printf("\n=========================== test1 ===========================\n");
	float input1[4] = {1.0f, 2.0f, 3.0f, 4.0f};
	float input2[4] = {2.0f, 3.0f, 4.0f, 5.0f};
	float output[4] = {0.0f};
	__m128 a = _mm_load_ps(input1);
	__m128 b = _mm_load_ps(input2);
	// __m128 c = a + b;
	__m128 c = _mm_mul_ss(a, b);
	_mm_storer_ps(output, c);
	for (int i=0; i<4; i++) {
		printf("%f, ", output[i]);
	}
}

int main(int argc, char* argv[]) {
	int ARRAY_SIZE = 1024 * 16;
	if (argc == 2) {
		ARRAY_SIZE = 1024 * atoi(argv[1]);
	}
	// test1();
	// test2(ARRAY_SIZE);
	test_sse(ARRAY_SIZE);
	
	return 0;
}



// __m128i vsuml = _mm_set1_epi32(0);
// __m128i vsumh = _mm_set1_epi32(0);
// __m128i vsum;
// int sum;

// for (int i = 0; i < N; i += 16)
// {
//     __m128i v = _mm_load_si128(&x[i]);
//     __m128i vl = _mm_unpacklo_epi8(v, _mm_set1_epi8(0));
//     __m128i vh = _mm_unpackhi_epi8(v, _mm_set1_epi8(0));
//     vsuml = _mm_add_epi32(vsuml, _mm_madd_epi16(vl, _mm_set1_epi16(1)));
//     vsumh = _mm_add_epi32(vsumh, _mm_madd_epi16(vh, _mm_set1_epi16(1)));
// }
// // do horizontal sum of 4 partial sums and store in scalar int
// vsum = _mm_add_epi32(vsuml, vsumh);
// vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 8));
// vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
// sum = _mm_cvtsi128_si32(vsum);

