# title
- Graphene: An IR for Optimized Tensor Computations on GPUs

# Author
- Bastian Hagedorn; Bin Fan; Hanfeng Chen; Cris Cecka; Michael Garland; Vinod Grover

# Institution
- NVIDIA

# Submitted
- ASPLOS ’23, March 25–29, 2023, Vancouver, BC, Canada
- ACM ISBN 978-1-4503-9918-0/23/03

# tags
- Software and its engineering → Parallel programming languages; Compilers; Software performance
- Intermediate Representation, Optimization, Tensor Computations, GPU, Compiler, Deep Learning, Code Generation

# Abstract
- Modern GPUs accelerate computations and data movements of multi-dimensional tensors in hardware. However, expressing optimized tensor computations in software is extremely challenging even for experts. Languages like CUDA C++ are centered around flat buffers in one-dimensional memory and lack reasonable abstractions for multi-dimensional data and threads. Existing tensor IRs are not expressive enough to represent the complex data-to-thread mappings required by the GPU tensor instructions.
- In this paper, we introduce Graphene, an intermediate representation (IR) for optimized tensor computations on GPUs. Graphene is a low-level target language for tensor compilers and performance experts while being closer to the domain of tensor computations than languages offering the same level of control such as CUDA C++ and PTX. In Graphene, multi-dimensional data and threads are represented as first-class tensors. Graphene’s tensors are hierarchically decomposable into tiles allowing to represent optimized tensor computations as mappings between data and thread tiles.
- We evaluate Graphene using some of the most important tensor computations in deep learning today, including GEMM, Multi-Layer Perceptron (MLP), Layernorm, LSTM, and Fused Multi-Head Attention (FMHA). We show that Graphene is capable of expressing all optimizations required to achieve the same practical peak performance as existing library implementations. Fused kernels beyond library routines expressed in Graphene significantly improve the end-to-end inference performance of Transformer networks and match or outperform the performance of cuBLAS(Lt), cuDNN, and custom handwritten kernels.

# Problem
- expressing optimized tensor computations extremely challenging

# Related work
- Fireiron 
- Mesh TensorFlow, GShard, GSPMD, P^2
- TVM, MLIR, and AMOS
- Marvel and MAESTRO 
- TVM (using the tensorize primitive), Diesel(polyhedral compiler) and UNIT
- Lift(can not generate TCU), Rise(limited support TCU) and Triton(highly complex compiler transformation passes, requires both initimate knowledge of the targeted GPU architectures as well as the compiler implementation)

# Key contribution
- serve as an alternative target language to CUDA C++/PTX that is closer to the domain of tensor computations

# Strengths
- express complex layouts are required to use the hardware as efficiently as possible
- data structure
  - Tensor = Name : Shape . ElementType . Memory
  - explicit shape
  - flexible hierarchy
- Graphene IR can integrated into machine learning compilers like XLA or TVM
- enables the specification of hierarchical meshes/threads

# Weaknesses
- dynamic shape
- out-of-bounds accesses

# Contents
## 1 INTRODUCTION
- expressing optimized tensor computations extremely challenging
  - vendor library kernels
  - higher-level built-ins operating
    - TVM’s tensorize primitive
    - MLIR’s gpu dialect
    - lack abstractions for complex data-to-thread mappings and layouts
  - built-in transformation passes
    - Triton’s instruction selection
- GEMM + pointwise epilogues, MLP, LSTM, Layernorm, and FMHA
## 2 OPTIMIZED GPU DATA MOVEMENTS
- ldmatrix
  - a warp (32 threads)
  - from shared memory into registers
  - performance drops 17%
  - matrics: 8x8 fp32, 16×16 fp16
- The 8 fp16 shared memory values accessed per thread
  ```
  00000000 88888888
  11111111 99999999
  22222222 ...
  ...      ...
  77777777 ffffffff

  16....16 24....24
  17....17 25....25
  ...      ...
  23....23 31....31
  ```
- The 8 values each thread receives in its registers
  ```
   0 0 1 1 2 2 3 3   4 4 5 5 6 6 7 7
   4 4 5 5 6 6 7 7   4 4 5 5 6 6 7 7
  ...               ...
  2828292930303131  2828292930303131

   0 0 1 1 2 2 3 3   4 4 5 5 6 6 7 7
   4 4 5 5 6 6 7 7   4 4 5 5 6 6 7 7
  ...               ...
  2828292930303131  2828292930303131
  ```
- CUDA C++/PTX data movement using ldmatrix
  ```
    // Reshape the warp into 2x2 groups of 8 threads
    int thr_grp_m = ((threadIdx.x / 16) % 2);
    int thr_grp_n = ((threadIdx.x / 8) % 2);
    int grp_local_idx = (threadIdx.x % 8); 
    // Tile and map shared mem (see a) - also convert pointer
    uint32_t src_ptr;
    void * dst_ptr = (void *) &smem[(thr_grp_m * 128 + thr_grp_n * 8 + grp_local_idx * 16)];
    asm volatile ("{ .reg .u64 src_ptr; \
                    cvta.to.shared.u64 src_ptr, %1; \
                    cvt.u32.u64 %0, src_ptr; }\n"
        : "=r"(src_ptr)
        : "l"(dst_ptr));
    // Each thread receives 8 vals (2 vals per 8x8 tile) - see b 
    int *dst = (int *)&regs[0]; // dst[x] points to two fp16 vals
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 \
                    {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(src_ptr));
  ```
- Graphene data movement using ldmatrix
  ```
    // Tensors(Sec.3)
    %1:[16, 16].fp16.SH // src
    %2:[ 2, 4].fp16.RF // dst
    // Logical thread groups(Sec.4)
    #3:[ 1].block
    #4:[32].thread
    %2 <- Move<<<#3, #4>>>(%1) { // Decomposable Specs (Sec.5)
          // Reshape the warp into 2�2 groups of 8 threads
          #5: [4].[8].thread = #4.tile(8)
          #6:[2,2].[8].thread = #5.reshape(0, [2,2])
          (@thr_grp_m, @thr_grp_n), @grp_local_idx = #6.indices()
          
          // Tile and map shared mem (see a) one 1�8 row per thread. 
          %7:[2, 2].[8, 8].fp16.SH = %1.tile([8, 8]) 
          %8:       [8, 8].fp16.SH = %7[@thr_grp_m, @thr_grp_n]
          %9:[8, 1].[1, 8].fp16.SH = %8.tile([1, 8]) 
          %10:      [1, 8].fp16.SH = %9[@grp_local_idx, 0]
          
          // Each thread receives 8 vals - 2�2 * 1�2 (see b) 
          %11:[2, 2].[1, 2].fp16.RF = %2.tile([1, 2]) 
          %11 <- Move<<<#3, #4>>>(%10) }
  ```

## 3 THE SHAPE OF TENSORS TO COME
- inspired by and builds upon NVIDIA’s CuTe programming model
- 3.1 Expressing Tensors in Graphene
  - Tensor = Name : Shape . ElementType . Memory
    - Name = %string
    - Shape = [Dims \n Stride] // ([dims:stride] in listings)
      - Dims = Stride = IntTuple
        - IntTuple = (Size, ..., Size)
          - Size = IntExpr | IntTuple
            - IntExpr = int | var | (IntExpr BinOp IntExpr)
              - BinOp = + | - | * | / | ...
    - ElementType = ScalarType | Shape . ElementType
      - ScalarType = fp16 | fp32 | i32 | ...
    - Memory = GL (global)| SH (shared)| RF (registers)
- 3.2 Memory Layouts
  - A two-dimensional colum-major tensor
    ```
    0 4 8  12 16 20 24 28
    1 5 9  13 17 21 25 29
    2 6 10 14 18 22 26 30
    3 7 11 15 19 23 27 31
    ```
    - A[0,3]=12
    - A:[4,8 \n 1,4].FP32
      - A[0,0]->A[1,0], stride=1
      - A[3,0]->A[0,1], stride=4
  - A two-dimensional row-major tensor
    ```
     0  1  2  3  4  5  6  7
     8  9 10 11 12 13 14 15
    16 17 18 19 20 21 22 23
    24 25 26 27 28 29 30 31
    ```
    - B[0,3]=3
    - B:[4,8 \n 8,1],FP32
  - hierarchical dimensions
    - A two-dimensional tensor with a more complex layout
      ```
      0  1  8  9 16 17 24 25
      2  3 10 11 18 19 26 27
      4  5 12 13 20 21 28 29
      6  7 14 15 22 23 30 31
      ```
      - C[0,3]=C[0,(1,1)]=9
        - We can still access with 2D coords
      - C:[4,(2,4) \n 2,(1,8)].FP32
    - Yet another way to lay out elements in memory
      ```
      0  2  8 10 16 18 24 26
      1  3  9 11 17 19 25 27
      4  6 12 14 20 22 28 30
      5  7 13 15 21 23 29 31
      ```
      - D[0,3]=D[(0,0),(1,1)]=10
      - D:[(2,2),(2,4) \n (1,4),(1,8)].FP32
  - bank conflict: multiple threads try to access different values stored in the same bank
- 3.3 Tiling Tensors
  - Regular contiguous tiles
  - Non-contiguous tiles
  - A 32-element column-major floating point tensor
    ```
     0  4  8 12 16 20 24 28
     1  5  9 13 17 21 25 29
     2  6 10 14 18 22 26 30
     3  7 11 15 19 23 27 31
    ```
    - A:[4,8 \n 1,4].FP32
  - A tiled tensor containing 4 tiles with 8 elements each
    ```
     0  4  8 12  16 20 24 28
     1  5  9 13  17 21 25 29

     2  6 10 14  18 22 26 30
     3  7 11 15  19 23 27 31
    ```
    - tile[2 \n 1, 4 \n 1]
    - B:[2,2 \n 2,16].[2,4 \n 1,4].FP32
      - 2, 16: offset in scalar elements to advance to the next tile
  - Creating 2x4 shaped tiles, interleaved in the 1st dim
    ```
     0  4  8 12  16 20 24 28
     
     1  5  9 13  17 21 25 29

     2  6 10 14  18 22 26 30

     3  7 11 15  19 23 27 31
    ```
      -  0  4  8 12;  2  6 10 14
      - 16 20 24 28; 18 22 26 30
      -  1  5  9 13;  3  7 11 15
      - 17 21 25 29; 19 23 27 31
    - tile(2 \n 2, 4 \n 1)
      - stride 2: every other row
    - C:[2,2 \n 1,16].[2,4 \n 2,4].FP32
      - +1 to advance to the next tile, +2 to advance to next tile-element
  - Creating 2x4 tiles that are non-contiguous in both dims
    ```
     0  4   8 12    16 20  24 28
     
     1  5   9 13    17 21  25 29

     2  6  10 14    18 22  26 30

     3  7  11 15    19 23  27 31
    ```
      -  0  4; 16 20;  2  6; 18 22
      -  8 12; 24 28; 10 14; 26 30
      -  1  5; 17 21;  3  7; 19 23
      -  9 13; 25 29; 11 15; 27 31
    - tile([2 \n 2], [2,2 \n 1,4])
    - D:[2,2 \n 1,8].[2,(2,2) \n 2,(4,16)].FP32
- 3.4 Parametric Shapes and Partial Tiles
## 4 LOGICAL THREAD GROUPS
- Representing the thread arrangment required for the ldmatrix instruction in Graphene: A warp is tiled into four groups arranged, as 2 × 2, of eight contiguous threads
  - A warp is represented as a 1D tensor of 32 contiguous threads
    - #warp:[32 \n 1].thread
    - int tidx = threadIdx.x % 32
  - Tiling a warp into four groups of 8 contiguous threads
    - #groups = #warp.tile([8:1])
    - $groups:[4 \n 8].[8 \n 1].thread
    - int grp_idx = threadIdx.x / 8
    - int tidx    = threadIdx.x % 8
  - Rearranging the groups two-dimensionally. Thread tensor tiles can have varying ranks
    - #groups2d = #groups.reshape(depth=0,[2,2 \n 2,1])
    - #groups2d:[2,2 \n 16,8].[8 \n 1].thread
    - int grp_m = (threadIdx.x / 16) % 2
    - int grp_n = (threadIdx.x / 8) % 2
    - int tidx    = threadIdx.x % 8
- Representing non-contiguous quad-pairs required for Volta’s Tensor Core instructions as logical thread groups
  - #quad_pairs = #warp .tile(4,2 \n 1,16)
  - #quad_pairs:[4 \n 4].[4,2 \n 1,16].thread
  - int qp_idx = (threadIdx.x / 4) % 4
  - int quad_idx = threadIdx.x % 4
  - int qp_hi_lo = (threadIdx.x / 16) % 2

## 5 SPECIFICATIONS AND DECOMPOSITIONS
- 5.1 Specifications for Expressing Computations
  - atomic spec encapsulate a self-contained block of computation
  - %out <- Spec<<<#blocks, #threads>>>(%in0,...) { optional decomposition }
    - %out: The output tensor(s)
    - Spec: The kind of computation: generic (Spec) or built-in (see Table 1)
    - #blocks, #threads: The execution configuration
    - %in0: The input tensors
    - optional: An optional implementation
- 5.2 Built-in Specifications
  - Graphene’s built-in specifications
    - Spec              Description
    - Move              Data Movements
    - MatMul            Matrix Multiplications (e.g., Tensor Cores, FMA)
    - UnaryPointwise    Elementwise unary computations (e.g., exp)
    - BinaryPointwise   Elementwise binary computations (e.g., add)
    - Reduction         Reduce tensor along one or more axes
    - Shfl              Exchange tensor values within thread groups
    - Init              Uniformly assign scalar value to a tensor
    - Allocate          Introduce new temporary data tensor
  - Examples for atomic specifications in Graphene and the associated (PTX) instructions they will be lowered into
    ```
      Spec        Threads     Inputs Outputs Associated Instruction
      Move        [1].thread  [].fp32.GL [].fp32.RF ld.global.u32
      Move        [1].thread [8].fp16.GL [8].fp16.RF ld.global.v4.u32
      Move        [1].thread [4].fp32.RF [4].fp32.SH st.shared.v4.u32
      Move        [32].thread [1,8].fp16.SH [2,2].[1,2].fp16.RF ld.matrix.sync.aligned.m8n8.x4.shared.b16
      BinaryPW<*> [1].thread [].fp16.RF, [].fp16.RF [].fp16.RF __hmul
      BinaryPW<+> [1].thread [2].fp16.RF, [2].fp16.RF [2].fp16.RF __hadd2
      MatMul      [1].thread [].fp16.RF, [].fp16.RF [].fp16.RF __hfma
      MatMul      [1].thread [2].fp16.RF, [2].fp16.RF [2].fp16.RF __hfma2
      MatMul      [1].thread [].fp32.RF, [].fp32.RF [].fp32.RF __fmaf
      MatMul      [(4,2) \n (1,16)].thread [4,1].fp16.RF,[1,4].fp16.RF [2,4].fp32.RF mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
      MatMul      [32].thread [2,2].[1,2].fp16.RF,[2,1].[2.1].fp16.RF [2,1].[1,2].fp32.RF mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    ```
- 5.3 Representing Fused Kernels
  - Generic specs describe the required input and output tensors as well as the participating threads executing this computation
- 5.4 Example: A Simple GEMM Kernel
  - Graphene IR
  ```
    %1:[1024, 1024].fp16.GL // omit strides
    %2:[1024, 1024].fp16.GL
    %3:[1024, 1024].fp16.GL
    #4:[ 8, 8].block : // omit strides too
    #5:[16, 16].thread
    %3 <- Spec<<<#4, #5>>>(%1, %2) {
      @bid_m, @bid_n = #4.indices()
      @tid_m, @tid_n = #5.indices()
      for(k=0; k < 1024; k += 1) {
        for(m=0; m < 8; m += 1) {
          for(n=0; n < 8; n += 1) {
            %6:[8, 1].[ 128, 1024].fp16.GL = %1.tile([128, _])
            %7:[1, 8].[1024, 128].fp16.GL = %2.tile([ _, 128])
            %8:[8, 8].[ 128, 128].fp16.GL = %3.tile([128, 128])
            // Assign tiles to blocks
            %9 :[ 128, 1024].fp16.GL = %6[@bid_m, 0]
            %10:[1024, 128].fp16.GL = %7[ 0, @bid_n]
            %11:[ 128, 128].fp16.GL = %8[@bid_m, @bid_n]
            // Tile for threads
            %12:[16, 1].[ 8, 1024].fp16.GL = %9.tile([8, _])
            %13:[ 1, 16].[1024, 8].fp16.GL = %10.tile([_, 8])
            %14:[16, 16].[ 8, 8].fp16.GL = %11.tile([8, 8])
            // Assign tiles to threads
            %15:[ 8, 1024].fp16.GL = %12[@tid_m, 0]
            %16:[1024, 8].fp16.GL = %13[ 0, @tid_n]
            %17:[ 8, 8].fp16.GL = %14[@tid_m, @tid_n]
            // Access scalars
            %18:[].fp16.GL = %15[m, k]
            %19:[].fp16.GL = %16[k, n]
            %20:[].fp16.GL = %17[m, n]
            // Target __hfma -> instruction is executed per thread
            #21:[].block = #4.scalar()
            #22:[].thread = #5.scalar()
            %20 <- MatMul<<<21, #22>>>(%18, %19)
    }}}}
  ```
  - Generated CUDA C++
  ```
    __global__ void
      graphene_kernel(const half *__restrict__ A, 
                      const half *__restrict__ B,
                            half *__restrict__ C) {
      int bid_m = (blockIdx.x % 8);
      int bid_n = ((blockIdx.x / 8) % 8);
      int tid_m = (threadIdx.x % 16);
      int tid_n = ((threadIdx.x / 16) % 16);
      #pragma unroll
      for (int k = 0; k < 1024; k += 1) {
      #pragma unroll
        for (int m = 0; m < 8; m += 1) {
      #pragma unroll 
          for (int n = 0; n < 8; n += 1) {
            C[(((bid_m * 128) + (bid_n * 131072)) + (((tid_m * 8) + (tid_n * 8192)) +  (m + (n * 1024))))] =
            __hfma(A[((bid_m * 128) + ((tid_m * 8) + (m + (k * 1024))))],
                   B[((bid_n * 131072) + ((tid_n * 8192) + (k + (n * 1024))))],
                   C[(((bid_m * 128) + (bid_n * 131072)) + (((tid_m * 8) + (tid_n * 8192)) + (m + (n * 1024))))]);
    }}}}
  ```
- 5.5 Code Generation
  - Graphene IR
    - specs
    - tensor manipulations
    - control-flow like loops and conditionals, synchronizations and barriers
    - other expressions not involving tensors
  - decomposed specs
    - implementation recursively 
    - build ASTs and compile those into thread index and buffer access expression
## 6 EVALUATION
- environment
  - V100 (SM70, Volta architecture)
    - M=N=5120,K=2048, M=N=128,K=32
  - RTX A6000(SM86, Ampere architecture).
    - M=N=5376,K=2048, M=N=128,K=32
  - CUDA-11.7, cuBLAS(Lt) version 11.10 and driver version 510.68.02.
  - NVIDIA’s Nsight-Compute profiler
  - FP16, FP32 Tensor Core accumulation
- Hypothesis A: Graphene can represent kernels that compete with high-performance library implementations
  - capable of expressing all optimizations necessary
  - large enough and evenly divides work among the available SMs, use exactly the same tile sizes as those used by cuBLAS 
- Hypothesis B: Graphene generates competitive fused kernels for important deep learning tensor computations
  - MLP
    - Graphene outperforms cuBLASLt by up to 2.39x
    - GEMM + bias addition + pointwise activation
    - fusion 20 layers
  - LSTM
    - LSTM-cell
      - using ReLU instead of tanh 
    - fuses all nodes into a single kernel
    ```
      Wt  Xt
       \  /
       GEMM
        | 
        | Wh Ht-1
        |  \  /
        |  GEMM
        | /
        add
        |  bias
        | /
        add
        |
        relu
    ```
  - Layernorm
    - a combination of pointwise and reduction computations
    ```
    T
    |  \
    |  mean
    |  /
    sub
    | \
    | square
    |  |
    | mean
    |  |  bias
    |  | /
    | add
    |  |
    | sqrt
    | /
    div
    |  B
    | /
    mul
    |  Y
    | /
    add
    |
    output
    ```
  - FMHA
    - 59% speedup
    - Fused Multi-Head Attention 
    - two back-to-back GEMMs with a softmax computation in between
    - softmax = two reductions and several pointwise operations
    - bert: 16 heads, batch-size of 32, hidden size 64, and sequence length 384
    - albert bert roberta distilroberta xlm-roberta
    ```
    Q   K
     \ /
     GEMM
      |  M
      | /
      mask
      |\
      | max
      | /
      sub
      |
      exp
      | \
      | sum
      | /
      div
      |  V
      | /
      GEMM
    ```

## 8 CONCLUSION

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights