# title
- CudaDMA: Optimizing GPU Memory Bandwidth via Warp Specialization

# Author & Institution
- Michael Bauer (Stanford) 
- Henry Cook (UC Berkeley)
- Brucek Khailany (NVIDIA Research)

# Submitted
- 12-18 November 2011
- https://dl.acm.org/doi/10.1145/2063384.2063400

# tags

# Abstract
- As the computational power of GPUs continues to scale with Moore's Law, an increasing number of applications are becoming limited by memory bandwidth. We propose an approach for programming GPUs with tightly-coupled specialized DMA warps for performing memory transfers between on-chip and off-chip memories. Separate DMA warps improve memory bandwidth utilization by better exploiting available memory-level parallelism and by leveraging efficient inter-warp producer-consumer synchronization mechanisms. DMA warps also improve programmer productivity by decoupling the need for thread array shapes to match data layout. To illustrate the benefits of this approach, we present an extensible API, CudaDMA, that encapsulates synchronization and common sequential and strided data transfer patterns. Using CudaDMA, we demonstrate speedup of up to 1.37x on representative synthetic microbenchmarks, and 1.15x-3.2x on several kernels from scientific applications written in CUDA running on NVIDIA Fermi GPUs.

# Problem
- programming GPU so challenging
  - Explicit data movement through memory hierarchy
  - Difficult to overlap computation and memory accesses

# Related work

# Key contribution

# Strengths

# Weaknesses

# Contents
## Overview of GPU Architecture
- GPU Architecture/Programming
  - on-chip memory
    - SM(stream multi-processor)
      - shared memory
      - CTA
  - off-chip DRAM
    - data
- Warp Definition
  - Each CTA is decomposed into warps
    - warp0, warp1, warp2, warp3, ...
    - A warp is 32 contiguous threads in the same CTA
  - SM performs scheduling at warp-granularity
    - Each warp has its own program counter
    - All threads in a warp execute in lock-step
    - Intra-warp divergence has performance penalty
    - Inter-warp divergence has no performance penalty
## Motivating Benchmark
- Modified SAXPY kernel, staging data through shared
  - Variable amount of arithmetic
  - Fixed amount of data transferred and number of warps
- graph
  - B/FLOP: increasing compute intensity
  - GB/s: IPC limit
- GPU Performance Challenges
  - Memory System Bottlenecks
    - Instruction Issue
      - Memory Level Parallelism(MLP)
    - Data Access Patterns
      - Coalescing
  - Computational Bottlenecks
    - Long-latency memory accesses
    - Synchronization overheads
  - Data Access Patterns
    - Control Divergence
  - Goal: remove entanglement between the bottlenecks
- GPU Programmability Challenges
  - Mismatch CTA size/shape and shared data size/shape
    - Leads to thread divergence (lots of ‘if’ statements)
  - Goal: decouple CTA size/shape from data size/shape
- Warp Specialization
  - Differentiate warps into compute and DMA*
  - DMA warps
    - Maximize MLP(Memory Level Parallelism)
  - Compute warps
    - No stalls due to memory
  - Producer-consumer synchronization
    - Enable better overlapping of compute and memory accesses
  - CudaDMA objects to manage warp specialization
    - Describe data transfer patterns
    - Independent of warp count
  - D. Merrill and A. Grimshaw. Revisiting Sorting for GPGPU Stream Architectures.

## CudaDMA API
- API
  - Declare CudaDMA object to manage shared buffer
  - Separate DMA and compute warps
  - Provide synchronization primitives
  - Perform repeated transfer operations
  - code
    ```
      class cudaDMA
      {
      public:
        // Base constructor
        __device__ cudaDMA (
          const int dmaID,
          const int num_dma_threads,
          const int num_comp_threads,
          const int thread_idx_start);
        public:
          __dev ice__ bool owns_this_thread();
        public:
          // Compute thread sync functions
          __device__ void start_async_dma();
          __device__ void wait_for_dma_finish();
        public:
          // DMA thread sync functions
          __device__ void wait_for_dma_start();
          __device__ void finish_async_dma();
        public:
          __device__ void execute_dma(void *src_ptr, void *dst_ptr);
      };
    ```
- CudaDMA Application Structure
  - Declare shared buffer at kernel scope
  - Declare CudaDMA object to manage buffer
  - Split DMA warps from compute warps
  - Load buffer using DMA warps
  - Process buffer using compute warps
  - Iterate (optional)
  - code
  ```
    __global__ void cuda_dma_kernel(float *data)
    {
      __shared__ float buffer[NUM_ELMTS];
      cudaDMA dma_ld(0, NUM_DMA_THRS, NUM_COMPUTE_THRS, NUM_COMPUTE_THRS);
      if (dma_ld.owns_this_thread()) {
        // DMA warps
        for (int i=0; i<NUM_ITERS; i++) {
          dma_ld.wait_for_dma_start();
          dma_ld.execute_dma(data,buffer);
          dma_ld.finish_async_dma();
        }
      }
      else { // Compute warps
        for (int i=0; i<NUM_ITERS; i++) {
          dma_ld.start_async_dma();
          dma_ld.wait_for_dma_finish();
          process_buffer(buffer);
        }
      }
    }
  ```
- Execution Model
  - Use PTX named barriers
    - bar.sync
    - bar.arrive
    - Available on Fermi
  - Fine-grained synchronization
  - graph
  ```
                              compute warps           DMA warps
                                | start_async_dma       | wait_for_dma_start
                                | bar.arrive            | bar.sync  
                                |                      \|/      
    named barrier 1   ----------------------------------------------------
                                | wait_for_dma_finish   | finish_async_dma
                                | bar.sync              | bar.arrive        
                               \|/                      |
    named barrier 2   ----------------------------------------------------
                                |                       |                iteration i
                                |                       |
                                |                      \|/
    named barrier 1   ----------------------------------------------------
                                |                       |                
                                |                       |       
                               \|/                      |
    named barrier 2   ---------------------------------------------------- 
                                                                         iteration i+1 
  ```

## Methodology
- Buffering Techniques
  - Usually one set of DMA warps per buffer
  - Single-Buffering
    - One buffer, one warp group
      ```
                                compute warps           DMA warps
                                  compute                | wait for
                                  on buffer A            | computation
                                                        \|/
      synchronization   ----------------------------------------------------
                                  | wait for             transfer into 
                                  | transfer             buffer A
                                \|/
      synchronization   ----------------------------------------------------
      ```
  - Double-Buffering
    - Two buffers, two warp groups
      ```
                                  compute warps           DMA warps buffer A        DMA warps buffer B
                                    compute                | wait for                 transfer into
                                    on buffer A            | computation              buffer B
                                                          \|/
        synchronization A -------------------------------------------------------------------------------------------
                                    compute                transfer into            | wait for 
                                    on buffer B            buffer A                 | computation
                                                                                   \|/
        synchronization B -------------------------------------------------------------------------------------------
      ```
  - Manual Double-Buffering
    - Two buffers, one warp group
      ```
                                compute warps           DMA warps
                                  compute                transfer into 
                                  on buffer A            buffer B
      synchronization A  ----------------------------------------------------
                                  compute                transfer into 
                                  on buffer B            buffer A
      synchronization B  ----------------------------------------------------
      ```
- CudaDMA Instances
  - CudaDMASequential
    - ------------XXXXXXXX------------
  - CudaDMAStrided
    - ----XXXX----XXXX----XXXX----XXXX-
  - CudaDMAIndirect
    - Arbitrary accesses
    - ----XXXX---XXXX-XXXX----XXXX----XXXX--XXXX-
  - CudaDMAHalo
    - 2D halo regions
    - XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    - XXXX---------------------------------XXXX
    - XXXX---------------------------------XXXX
    - XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
  - CudaDMACustom
- Access Patterns
  - Explicitly state data loading pattern in code
  - Decouple implementation from transfer pattern
  - Common patterns implemented by experts
    - Used by application programmers
  - Optimized for high memory bandwidth at low warp count

## Experiments
- Micro-Benchmarks
  - Same modified SAXPY kernel shown earlier
  - Fix compute intensity (6 B/FLOP), vary warp count
- BLAS2: SGEMV
  - Dense matrix-vector multiplication
  - CudaDMASequential for loading vector elements
  - CudaDMAStrided for loading matrix elements
  - Varied buffering schemes
  - Up to 3.2x speedup
- 3D Finite Difference Stencil
  - 8th order in space, 1st order in time computation
  ```
    y /|\
       |
       | //\ z
       | /
       |/____________\
                     /  x
  ```
  - Load 2D slices into shared for each step in Z-dimension
  - Loading halo cells uses uncoalesced accesses
    - Earlier version of cudaDMAHalo
  - Figures from: P. Micikevicius. 3D Finite Difference Computation on GPUs Using CUDA
- 3D Finite-Difference Stencil
  - Use DMA warps for loading halo cells as well as main block cells
  - Speedups from 13-15%
  - Improvement from more MLP and fewer load instructions
## Conclusions
- CudaDMA
  - Extensible API
  - Create specialized DMA Warps
  - Works best for moderate compute intensity applications
  - Decouple transfer pattern from implementation
- Optimized instances for common patterns
  - CudaDMASequential, CudaDMAStrided
  - CudaDMAIndirect, CudaDMAHalo
- Speedups on micro-benchmarks and applications
- Download CudaDMA:
  - http://code.google.com/p/cudadma
  - http://lightsighter.github.io/CudaDMA/
  - Tech Talk at NVIDIA Booth on Thursday at 1pm

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights