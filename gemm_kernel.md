# reference
- https://zhuanlan.zhihu.com/p/631227862
- https://github.com/KnowingNothing/MatmulTutorial
- https://dl.acm.org/doi/pdf/10.1145/3582016.3582018

# 1.baseline
## CuBLAS
  ```
  // hgemm
  inline cublasStatus_t
  gemm(cublasHandle_t handle,
      cublasOperation_t transA, cublasOperation_t transB,
      int m, int n, int k,
      const float* alpha,
      const half* A, int ldA,
      const half* B, int ldB,
      const float* beta,
      half* C, int ldC)
  {
    return cublasGemmEx(handle, transA, transB, m, n, k,
                        reinterpret_cast<const float*>(alpha),
                        reinterpret_cast<const __half*>(A), CUDA_R_16F, ldA,
                        reinterpret_cast<const __half*>(B), CUDA_R_16F, ldB,
                        reinterpret_cast<const float*>(beta),
                        reinterpret_cast<      __half*>(C), CUDA_R_16F, ldC,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  ```
  - FP16-FP32
  - row first
  - transA=true, transB=false
  - alpha=1，beta=0
  - Epilogue: Matrix constant multiply and add
## HW
  - A100 PCIE 80GB GPU
## profiling
- CUDA Event: 10 times hotwarm，200 times average, latency=0.755333ms, TFLOPS=156.726

# 2.Naive WMMA Kernel
## tile
- 1 threadblock use 2x2=4 warp; 1 warp=32 lane
- threadblock:128x128x32
  - blockXdim -> N
  - blockYdim -> M
- warp:64x64x16
  - threadXdim=32, threadYdim=2, threadZdim=2
  - threadYdim -> N
  - threadZdim -> M
- TensorCore
  - 1 wmma = 16x16x16
  - 1 warp = 64x64x16, (64x64x16)/(16x16x16) = 4*4 = 16
    - matrix_a: 4 fragment
    - matrix_b: 4 fragment
    - accumulator: 16 fragment
  ```
  const int MI = 128;
  const int NI = 128;
  const int KI = 32;
  const int MII = 64;
  const int NII = 64;
  const int KII = 16;
  const int wmmaM = 16;
  const int wmmaN = 16;
  const int wmmaK = 16;

  __global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
      // A is row-major
      // B is col-major
      // 128 threads [x, y, z] = [32, 2, 2]
      // threadblock mma: 128x128x32
      // warp mma: 64x64x16
      extern __shared__ uint8_t shared_storage[];
      half *SA = reinterpret_cast<half *>(shared_storage);
      half *SB = reinterpret_cast<half *>(shared_storage + MI*KI*sizeof(half)); // 128*32*2=8192
      float *SC = reinterpret_cast<float *>(shared_storage);

      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII/wmmaM]; // 64/16=4
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII/wmmaN]; // 64/16=4
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII/wmmaM*NII/wmmaN]; // 4*4=16

      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          nvcuda::wmma::fill_fragment(Accum[mii*(NII/wmmaN) + nii], 0.0);
        }
      }
      for (int ko = 0; ko < K/KI; ko += 1) {
        loadSmemA(SA, A, M, K, ko);
        loadSmemB(SB, B, N, K, ko);
        __syncthreads();
        for (int ki = 0; ki < KI/KII; ki += 1) {
          // 64x64x16 mma for each warp
          loadFragA(FragA, SA, ki);
          loadFragB(FragB, SB, ki);
          for (int mii = 0; mii < MII/wmmaM; mii += 1) {
            for (int nii = 0; nii < NII/wmmaN; nii += 1) {
              // 16x16x16 for each wmma
              nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
            }
          }
        }
      }
      storeAccum(SC, Accum);
      __syncthreads();
      storeSmemC(C, SC, M, N);
  }
  ```
- shared memory usage = MAX(SmemA+SmemB, SmemC)=MAX(8192+8192, 128*128*4)=MAX(16384, 65536)=65536/1000=65.54KB
## shared memory load
- mapping directly: load one data per thread
  ```
  __device__ void loadSmemA(half *smem, half *A, int M, int K, int ko) {
    // load 128*32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz*64 + ty*32 + tx;
    for (int i = 0; i < 32; ++i) {
      int row = i*4 + tid/32;
      int col = tid%32;
      // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]
      smem[row/16*(2*16*16) + col/16*(16*16) + row%16*16 + col%16] = A[(by*128 + row)*K + (ko*KI+col)];
    }
  }
  ```
## fragment load
- intrinsic
  ```
  __device__ void loadFragA(nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> *frag, half *smem, int ki) {
    // load 64x16
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
      int row = tz*64 + i*16;
      int col = ki*KII;
      nvcuda::wmma::load_matrix_sync(frag[i], smem + row/16*(2*16*16) + col/16*(16*16), 16);
    }
  }
  ```
## perf
- latency=4.46636ms，TFLOPS=26.5048

# 3.4-Stage Pipeline WMMA Kernel
## A100: cp.asyncs: load from global memory to shared memory
## 4-stage
  - SmemA = SmemA*4, SmemB=SmemB*4
  - shared memory usage = MAX(SmemA+SmemB, SmemC)=MAX(8192*4+8192*4, 128*128*4)=MAX(65536, 65536)=65536/1000=65.54KB
## pipeline
  ```
    // prologue
    g2s_a_0, g2s_b_0
             g2s_a_1, g2s_b_1
                      g2s_a_2, g2s_b_2
    // main_loop
                      cp.async.wait_group(2), s2l_a_0, s2l_b_0, tcu_0
                               g2s_a_3, g2s_b_3
    // epilogue
                               cp.async.wait_group(2), s2l_a_1, s2l_b_1, tcu_1
                                  s2g_0
                                  cp.async.wait_group(2), s2l_a_2, s2l_b_2, tcu_2
                                    s2g_1
                                    cp.async.wait_group(2), s2l_a_3, s2l_b_3, tcu_3
                                    s2g_2
                                    s2g_3
  ```
## code
```
__global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
  // A is row-major
  // B is col-major
  // 128 threads [x, y, z] = [32, 2, 2]
  // threadblock mma: 128x128x32
  // warp mma: 64x64x16
  extern __shared__ uint8_t shared_storage[];
  half *SA1 = reinterpret_cast<half *>(shared_storage);
  half *SA2 = SA1 + MI*KI;
  half *SA3 = SA2 + MI*KI;
  half *SA4 = SA3 + MI*KI;
  half *SB1 = SA4 + MI*KI;
  half *SB2 = SB1 + NI*KI;
  half *SB3 = SB2 + NI*KI;
  half *SB4 = SB3 + NI*KI;
  float *SC = reinterpret_cast<float *>(shared_storage);

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::row_major> FragA[MII/wmmaM];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, wmmaM, wmmaN, wmmaK, half, nvcuda::wmma::col_major> FragB[NII/wmmaN];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, wmmaM, wmmaN, wmmaK, float> Accum[MII/wmmaM*NII/wmmaN];

  for (int mii = 0; mii < MII/wmmaM; mii += 1) {
    for (int nii = 0; nii < NII/wmmaN; nii += 1) {
      nvcuda::wmma::fill_fragment(Accum[mii*(NII/wmmaN) + nii], 0.0);
    }
  }

  // prologue
  loadSmemA(SA1, A, M, K, 0);
  loadSmemB(SB1, B, N, K, 0);
  asm volatile("cp.async.commit_group;\n" ::);

  loadSmemA(SA2, A, M, K, 1);
  loadSmemB(SB2, B, N, K, 1);
  asm volatile("cp.async.commit_group;\n" ::);

  loadSmemA(SA3, A, M, K, 2);
  loadSmemB(SB3, B, N, K, 2);
  asm volatile("cp.async.commit_group;\n" ::);

  for (int ko = 0; ko < K/KI - 4; ko += 4) {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 3 < K/KI) {
      loadSmemA(SA4, A, M, K, ko + 3);
      loadSmemB(SB4, B, N, K, ko + 3);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA1, ki);
      loadFragB(FragB, SB1, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 4 < K/KI) {
      loadSmemA(SA1, A, M, K, ko + 4);
      loadSmemB(SB1, B, N, K, ko + 4);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA2, ki);
      loadFragB(FragB, SB2, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 5 < K/KI) {
      loadSmemA(SA2, A, M, K, ko + 5);
      loadSmemB(SB2, B, N, K, ko + 5);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA3, ki);
      loadFragB(FragB, SB3, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 6 < K/KI) {
      loadSmemA(SA3, A, M, K, ko + 6);
      loadSmemB(SB3, B, N, K, ko + 6);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA4, ki);
      loadFragB(FragB, SB4, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }
  }

  // the last 4 iterations
  {
    int ko = (K/KI/4 - 1)*4;
    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 3 < K/KI) {
      loadSmemA(SA4, A, M, K, ko + 3);
      loadSmemB(SB4, B, N, K, ko + 3);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA1, ki);
      loadFragB(FragB, SB1, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
    __syncthreads();
    if (ko + 4 < K/KI) {
      loadSmemA(SA1, A, M, K, ko + 4);
      loadSmemB(SB1, B, N, K, ko + 4);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA2, ki);
      loadFragB(FragB, SB2, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(1));
    __syncthreads();
    if (ko + 5 < K/KI) {
      loadSmemA(SA2, A, M, K, ko + 5);
      loadSmemB(SB2, B, N, K, ko + 5);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA3, ki);
      loadFragB(FragB, SB3, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }

    asm volatile("cp.async.wait_group %0;\n" ::"n"(0));
    __syncthreads();
    if (ko + 6 < K/KI) {
      loadSmemA(SA3, A, M, K, ko + 6);
      loadSmemB(SB3, B, N, K, ko + 6);
    }
    for (int ki = 0; ki < KI/KII; ki += 1) {
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA4, ki);
      loadFragB(FragB, SB4, ki);
      for (int mii = 0; mii < MII/wmmaM; mii += 1) {
        for (int nii = 0; nii < NII/wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          nvcuda::wmma::mma_sync(Accum[mii*(NII/wmmaN) + nii], FragA[mii], FragB[nii], Accum[mii*(NII/wmmaN) + nii]);
        }
      }
    }
  }
  storeAccum(SC, Accum);
  __syncthreads();
  storeSmemC(C, SC, M, N);
}

```
## loadSmemA
```
  __device__ void loadSmemA(half *smem, half *A, int M, int K, int ko) {
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 64 + ty * 32 + tx;
    for (int i = 0; i < 4; ++i) {
      int row = i * 32 + tid / 4;
      int col = tid % 4 * 8;
      // layout: [row_out, col_out, row_in, col_in] = [8, 2, 16, 16]

      void *ptr = (void *)(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
      uint32_t smem_ptr;

      asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
          : "=r"(smem_ptr)
          : "l"(ptr));

      asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                    "l"(&A[(by * 128 + row) * K + (ko * KI + col)]),
                  "n"(16));
    }
  }
```
## perf
- latency=1.15745ms，TFLOPS=102.277, 65.26% CuBLAS

# 4. 4-Stage Pipeline MMA Kernel
## change
- wmma -> mma; loadFrag, mmaSync, storeAccum
## loadFrag
- mma: special fragment layout
- mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
  - %laneid:{fragments}
  - MMA.m16n8k8 fragment layout for matrix A with .f16/.f32 type
  ```
  Row//Col  0 1         2 3         4 5         6 7
  0         T0:{a0,a1}  T1:{a0,a1}  T2:{a0,a1}  T3:{a0,a1}
  1         T4:{a0,a1}  T5:{a0,a1}  T6:{a0,a1}  T7:{a0,a1}
  ...
  7         T28:{a0,a1} T29:{a0,a1} T30:{a0,a1} T31:{a0,a1}
  8         T0:{a0,a1}  T1:{a0,a1}  T2:{a0,a1}  T3:{a0,a1}
  9         T4:{a0,a1}  T5:{a0,a1}  T6:{a0,a1}  T7:{a0,a1}
  ...
  15        T28:{a0,a1} T29:{a0,a1} T30:{a0,a1} T31:{a0,a1}
  ```
- g2s_a
  - SmemA[64x16]
  - (64*16)/(16*8) = 4*2 = 8 = 2*2 half
  - load 2 half = load 1 unsigned int
- 422 loop
  ```
  __device__ void loadFragA(unsigned int *frag, half *smem, int ki) {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) { // 64/16
      for (int j = 0; j < 2; ++j) { // 16/8
        for (int k = 0; k < 2; ++k) { // 2 half
          int row = tz*64 + i*16 + j*8 + tx/4;
          int col = ki*KII + k*8 + tx%4*2;
          unsigned int *ptr = reinterpret_cast<unsigned int *>(smem + row/16*(2*16*16) + col/16*(16*16) + row%16*16 + col%16);
          frag[i*4 + j*2 + k] = ptr[0];
        }
      }
    }
  }
  ```
## storeAccum
- MMA.m16n8k8 fragment layout for accumulator matrix C/D with .f16x2/.f32 type
  ```
  Row//Col  0 1         2 3         4 5         6 7
  0         T0:{c0,c1}  T1:{c0,c1}  T2:{c0,c1}  T3:{c0,c1}
  1         T4:{c0,c1}  T5:{c0,c1}  T6:{c0,c1}  T7:{c0,c1}
  ...
  7         T28:{c0,c1} T29:{c0,c1} T30:{c0,c1} T31:{c0,c1}
  8         T0:{c0,c1}  T1:{c0,c1}  T2:{c0,c1}  T3:{c0,c1}
  9         T4:{c0,c1}  T5:{c0,c1}  T6:{c0,c1}  T7:{c0,c1}
  ...
  15        T28:{c0,c1} T29:{c0,c1} T30:{c0,c1} T31:{c0,c1}
  ```
- Accum[64,64]
  - (64*64)/(16*8) = 4*8 = 32 = 2*2 float
- 4822 loop = 4422 loop
  ```
  __device__ void storeAccum(float *ptr, float *frag) {
    // frag [r, c, _]: [2, 2, 2]
    // store 64x64
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < 2; ++r) {
          for (int c = 0; c < 2; ++c) {
            int row = tz * 64 + i * 16 + r * 8 + tx / 4;
            int col = ty * 64 + j * 16 + c * 8 + tx % 4 * 2;
            float *dst = ptr + row / 16 * (8 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16;
            dst[0] = frag[i * 32 + j * 8 + r * 4 + c * 2];
            dst[1] = frag[i * 32 + j * 8 + r * 4 + c * 2 + 1];
          }
        }
      }
    }
  }
  ```
## mma.sync
- wmma::mma_sync -> mma:sync
- mma store:16x16x16
  - (16*16*16)/(16*8*8) = 4
- mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
  ```
  __device__ void mmaSync(unsigned int *fragA, unsigned int *fragB, float *accum)
  {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7,  %8,  %9,  %10};\n"
          : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
          : "r"(fragA[0]), "r"(fragA[2]),
            "r"(fragB[0]),
            "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7,  %8,  %9,  %10};\n"
          : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
          : "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[1]),
            "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7,  %8,  %9,  %10};\n"
          : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
          : "r"(fragA[0]), "r"(fragA[2]),
            "r"(fragB[2]),
            "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));

      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5},"
          "{%6},"
          "{%7,  %8,  %9,  %10};\n"
          : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
          : "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[3]),
            "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));
  }
  ```
## perf
- latency=1.33858ms，TFLOPS=88.4372

# 5. 4-Stage Pipeline MMA + ldmatrix Kernel
- ldmatrix: load from shared memory to register, 16x16 per time
## loadFrag
  ```
  __device__ void loadFragA(unsigned int *frag, half *smem, int ki): {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
      int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
      int col = ki * KII + tx / 8 % 2 * 8;
      void *ptr = (void *)(smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16);
      uint32_t smem_ptr;
      asm(
          "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
          : "=r"(smem_ptr)
          : "l"(ptr));
      asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                  : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]), "=r"(frag[i * 4 + 3])
                  : "r"(smem_ptr));
    }
  }
  ```
## perf
- latency=1.48608ms，TFLOPS=79.6593
## mmaSync
- mma store:16x16x16
  - (16*16*16)/(16*8*16) = 2
- mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  ```
  __device__ void mmaSync(unsigned int *fragA, unsigned int *fragB, float *accum) {
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(accum[0]), "=f"(accum[1]), "=f"(accum[4]), "=f"(accum[5])
          : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[0]), "r"(fragB[1]),
            "f"(accum[0]), "f"(accum[1]), "f"(accum[4]), "f"(accum[5]));

      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,  %1,  %2,  %3},"
          "{%4,  %5,  %6,  %7},"
          "{%8,  %9},"
          "{%10, %11, %12, %13};\n"
          : "=f"(accum[2]), "=f"(accum[3]), "=f"(accum[6]), "=f"(accum[7])
          : "r"(fragA[0]), "r"(fragA[2]), "r"(fragA[1]), "r"(fragA[3]),
            "r"(fragB[2]), "r"(fragB[3]),
            "f"(accum[2]), "f"(accum[3]), "f"(accum[6]), "f"(accum[7]));
  }
  ```
  - latency=1.47957ms，TFLOPS=80.0528

# 6. 4-Stage Pipeline MMA (+ldmatrix) + Simplify Kernel
- loadFrag optimize
  - simplify repeated scalar calculations, such as calculating row and col
  - 16x16 layout
  ```
  __device__ void loadFragA(unsigned int *frag, half *smem, int ki) {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    int row = tz * 64 + tx / 4;
    int col = ki * KII + tx % 4 * 2;
    half *ptr = smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16;
    for (int i = 0; i < 4; ++i) {
      frag[i * 4 + 0] = *(reinterpret_cast<unsigned int *>(ptr));
      frag[i * 4 + 1] = *(reinterpret_cast<unsigned int *>(ptr + 8));

      frag[i * 4 + 2] = *(reinterpret_cast<unsigned int *>(ptr + 8 * 16));
      frag[i * 4 + 3] = *(reinterpret_cast<unsigned int *>(ptr + 8 * 16 + 8));
      ptr += 16 * 16 * 2;
      }
  }
  ```
  - latency=1.34029ms，TFLOPS=88.3243
- do not use ldmatrix
  ```
  __device__ void loadFragA(unsigned int *frag, half *smem, int ki) {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    int row = tz * 64 + tx / 4;
    int col = ki * KII + tx % 4 * 2;
    half* ptr = smem + row / 16 * (2 * 16 * 16) + col / 16 * (16 * 16) + row % 16 * 16 + col % 16;
    for (int i = 0; i < 4; ++i) {
      frag[i * 4 + 0] = *(reinterpret_cast<unsigned int *>(ptr));
      frag[i * 4 + 1] = *(reinterpret_cast<unsigned int *>(ptr + 8));

      frag[i * 4 + 2] = *(reinterpret_cast<unsigned int *>(ptr + 8 * 16));
      frag[i * 4 + 3] = *(reinterpret_cast<unsigned int *>(ptr + 8 * 16 + 8));
      ptr += 16 * 16 * 2;
    }
  }
  ```
  - latency=1.21901ms，TFLOPS=97.1117
- It is difficult to gain performance by replacing it with mma instructions casually.

# 7. 4-Stage Pipeline MMA + Simplify + Double Threading Kernel
- increase warp in 1 block
- MNK = 421
- MNK = 222
  - rfactor: The partial sum needs to be accumulated synchronously between warps
- There is no significant improvement in performance

# 8. 4-Stage Pipeline MMA + Simplify + Double Threading + SMem Crosswise Kernel
## profile
- very low utilization of memory throughput -> shared memory bank conflict
## bank conflict during g2s
- 1 bank = 4 bytes
- 32 bank
- 
```
  global memory             shared memory
T0   T1   T2   T3           T0    T1
T4   T5   T6   T7           T4    T5
T8   T9   T10  T11          T8    T9
T12  T13  T14  T15          T12   T13      32 bank
T16  T17  T18  T19          T16   T17
T20  T21  T22  T23          T20   T21
T24  T25  T26  T27          T24   T25
T28  T29  T30  T31          T28   T29
                            T2    T3
                            T6    T7    
                            ...
```
- load 8 half per time, use 4 bank, 
- 8 threads will occupy 32 banks
- serial store: T0,T1,T4,T5,T8,T9,T12,T13
- gpu schedule: T0-T7
  - conflict between: T0&T2, T1&T3, T4&T6, T5&T7
## bank conflict during s2l
## solution
- padding 1
  - cp.async constrain: address needs align 16
- padding 16
  - increase shared memory size
- crosswise layout
  - For a replacement of the column address with XOR method
  - XOR the 3-5 bits of the column address according to the lowest three bits of the row address
    - because it is necessary to avoid 0-2 bits, which corresponds to The length of a load of 8 elements
  - col = col ^(((row &3)<<3))
## loadSmemA
  ```
  __device__ void loadSmemA(half *smem, half *A, int M, int K, int ko) {
    // load 128 * 32
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tz * 128 + ty * 32 + tx;
    for (int i = 0; i < 2; ++i) {
      int logic_row = i * 64 + tid / 4;
      int logic_col = tid % 4 * 8;
      int row = i * 32 + tid / 8;
      int col = tid % 8 * 8;
      col = col ^ (((row & 3) << 3));
      void *ptr = (void *)(smem + row * 64 + col);
      uint32_t smem_ptr;

      asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
          "%0, smem_ptr; }\n"
          : "=r"(smem_ptr)
          : "l"(ptr));

      asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(smem_ptr),
                  "l"(&A[(by * 128 + logic_row) * K + (ko * KI + logic_col)]),
                  "n"(16));
    }
  }
  ```
## loadFragA
  ```
  __device__ void loadFragA(unsigned int *frag, half *smem, int ki) {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          int row = tz * 64 + i * 16 + j * 8 + tx / 4;
          int col = ki * KII + k * 8 + tx % 4 * 2;
          col = row % 2 * 32 + col;
          row = row / 2;
          col = col ^ ((row & 3) << 3);
          unsigned int *ptr = reinterpret_cast<unsigned int *>(smem + row * 64 + col);
          frag[i * 4 + j * 2 + k] = ptr[0];
        }
      }
    }
  }
  ```
## perf
- latency=1.06895ms，TFLOPS=110.744
## ldmatrix
```
  __device__ __forceinline__ void loadFragA(unsigned int *frag, half *smem, int ki) {
    // frag: [j, k]: [2, 2]
    // load 64x16
    int tx = threadIdx.x;
    int tz = threadIdx.z;

  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = tz * 64 + i * 16 + tx / 16 * 8 + tx % 8;
      int col = ki * KII + tx / 8 % 2 * 8;
      col = row % 2 * 32 + col;
      row = row / 2;
      col = col ^ (((row & 3) << 3));
      void *ptr = (void *)(smem + row * 64 + col);
      uint32_t smem_ptr;
      asm("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
          "%0, smem_ptr; }\n"
          : "=r"(smem_ptr)
          : "l"(ptr));
      asm volatile(
          "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(frag[i * 4 + 0]), "=r"(frag[i * 4 + 1]), "=r"(frag[i * 4 + 2]),
            "=r"(frag[i * 4 + 3])
          : "r"(smem_ptr));
    }
  }
```
  - latency=0.00935ms，TFLOPS=106.942

# 9. 4-Stage Pipeline MMA + SMem Crosswise + Partial ldmatrix Kernel
## profiler
  - The performance of warp is not good
## fragment double buffer
  - no change in performance
## mixed use of ldmatrix
  - A not use ldmatrix，B use ldmatrix
  - Different instructions can be parallelized
  - latency=0.00754ms，TFLOPS=132.451 TFLOPS，84.51% CuBLAS
## unroll
- The outer layer is not expanded, and the inner layer is expanded to reduce the register pressure
## Snake traversal
- Snake traversal is added to increase the reuse of fragments
```
  __global__ void matmul(half *A, half *B, half *C, int M, int N, int K) {
    // A is row-major
    // B is col-major
    // 128 threads [x, y, z] = [32, 2, 2]
    // threadblock mma: 128x128x32
    // warp mma: 64x64x16
    extern __shared__ uint8_t shared_storage[];
    half *SA1 = reinterpret_cast<half *>(shared_storage);
    half *SA2 = SA1 + MI * KI;
    half *SA3 = SA2 + MI * KI;
    half *SA4 = SA3 + MI * KI;
    half *SB1 = SA4 + MI * KI;
    half *SB2 = SB1 + NI * KI;
    half *SB3 = SB2 + NI * KI;
    half *SB4 = SB3 + NI * KI;
    half *SA[] = {SA1, SA2, SA3, SA4};
    half *SB[] = {SB1, SB2, SB3, SB4};
    float *SC = reinterpret_cast<float *>(shared_storage);

    unsigned int FragA[4 * 4]; // [4, 4]
    unsigned int FragB[4 * 4]; // [4, 4]

    float Accum[4 * 4 * 8] = {0.0}; // [4, 4, 8]

    // prologue
    for (int i = 0; i < 3; ++i) {
      loadSmemA(SA[i], A, M, K, i);
      loadSmemB(SB[i], B, N, K, i);
      asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int ko = 0; ko < K / KI; ko += 1) {
      asm volatile("cp.async.wait_group %0;\n" ::"n"(2));
      __syncthreads();
      
      // 64x64x16 mma for each warp
      loadFragA(FragA, SA[(ko)%4], 0);
      loadFragB(FragB, SB[(ko)%4], 0);
  #pragma unroll
      for (int mii = 0; mii < MII / wmmaM; mii += 1) {
  #pragma unroll
        for (int nii = 0; nii < NII / wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
          mmaSync(&FragA[mii * 4], &FragB[n * 4], &Accum[mii * 32 + n * 8]);
        }
      }

      if (ko + 3 < K / KI) {
        loadSmemA(SA[(ko+3)%4], A, M, K, ko + 3);
        loadSmemB(SB[(ko+3)%4], B, N, K, ko + 3);
        asm volatile("cp.async.commit_group;\n" ::);
      }

      // 64x64x16 mma for each warp
      loadFragA(FragA, SA[(ko)%4], 1);
      loadFragB(FragB, SB[(ko)%4], 1);
  #pragma unroll
      for (int mii = 0; mii < MII / wmmaM; mii += 1) {
  #pragma unroll
        for (int nii = 0; nii < NII / wmmaN; nii += 1) {
          // 16x16x16 for each wmma
          int n = (mii & 1) ? NII / wmmaN - 1 - nii : nii;
          mmaSync(&FragA[mii * 4], &FragB[n * 4], &Accum[mii * 32 + n * 8]);
        }
      }
    }
    storeAccum(SC, Accum);
    __syncthreads();
    storeSmemC(C, SC, M, N);
  }
```
- latency=0.83905ms，TFLOPS=141.088，90.02% CuBLAS
