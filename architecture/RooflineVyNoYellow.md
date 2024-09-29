# title
Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures*
https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf

# author
Samuel Williams, Andrew Waterman, and David Patterson

# Institution
Berkeley

# ABSTRACT
We propose an easy-to-understand, visual performance model that offers insights to programmers and architects on improving parallel software and hardware for floating point computations

# key concepts
- Arithmetic Intensity (AI)
  This is defined as the number of floating-point operations (FLOPs) performed per byte of data transferred from memory. It serves as a crucial metric for determining how well an algorithm utilizes the available memory bandwidth.
- Performance Boundaries
  The model features two primary ceilings:
  - Peak Performance Ceiling(π)
    This represents the maximum achievable performance based on the hardware's capabilities, typically measured in GFLOPs.
  - Memory Bandwidth Ceiling(β)
    This indicates the maximum data transfer rate from memory, which can limit performance if not adequately managed.

  - P = min{π, β * I}
    P is the attainable performance,
    π is the peak performance,
    β is the memory bandwidth, and
    I is the arithmetic intensity

# Roofline Model


# 1.Intrudoction
caches, pipelining, superscalar instruction issue, and out-oforder execution

# 2.PERFORMANCE MODELS
Amdahl’s Law which states simply that the performance gain of a parallel computer is limited by the serial portion of a parallel program.

# 3.THE ROOFLINE MODEL
- off-chip memory bandwidth will often be the constraining resource
  HBM DRAM
  on-chip cache: register, L1, L2, shared
- measure traffic between the caches and memory rather than between the processor and the caches

- 2 hardware limits
  The x-axis represents arithmetic intensity (FLOPs/byte)
  The y-axis represents performance in GFLOPs
  BW = bytes per second
- Key Insights from the Model
  Performance Bottlenecks
    compute bound
    memory bound
  Optimization Potential
    If a workload is far below the roofline, it indicates significant room for optimization
    increasing operational intensity or improving data locality to enhance cache utilization
- Attainable GFlops/sec = Min(Peak Floating Point Performance, Peak Memory Bandwidth x Operational Intensity)

# 4.ADDING CEILINGS TO THE MODEL
- To reduce computational bottlenecks
    Improve instruction level parallelism (ILP) and apply SIMD
      loop_unroll
    balance operation mix
      LSU,TCU,BFU
      loop_split
- To reduce memory bottlenecks
    increase stride accesses
      loop_split, loop_reorder
    memory affinity
      s2l loop_split ?
      allocates data and the threads tasked to that data to the same memory-processor pair
    requires keeping many memory operations in flight
      double_buffer

# 5.Tying the 3Cs to Operational Intensity
  reduce traffic from conflict misses by padding arrays to change cache line addressing
    storage_aline, SME
  no-allocate store instruction

# 6.DEMONSTRATION OF THE MODEL
- 6.1 Four Diverse Multicore Computers
  Table 1. Characteristics of four recent multicores
    ISA
    total_threads
    total_cores
    total_sockets
    GHz
    peak_GFlop/s
    peak_DRAM_GB/s
    stream_GB/s
    DRAM_type
- 6.2 Four Diverse Floating-Point Kernels
  Phil Colella Seven Dwarfs
  Table 2. Characteristics of four FP Kernels.
    y = A@x
    Lattice-Boltzmann Magnetohydro dynamics
    A multigrid kernel that updates 7 nearby points in a 3-D stencil for a 256^3 problem
    Three-Dimensional Fast Fourier Transform (2 sizes: 128^3 and 512^3)
- 6.3 Roofline Models and Results
  Table 3. Kernel Optimizations
    1.Memory Affinity. Reduce accesses to DRAM memory attached to the other socket.
    2.Long unit-stride accesses. Change loop structures to generate long unit-stride accesses to engage the prefetchers. Also reduces TLB misses
    3.Software Prefetching. To get the most out of the memory systems, both software and hardware prefetching were used.
    4.Reduce conflict misses. Pad arrays to improve cache-hit rates.
    5.Unroll and Reorder Loops. To expose sufficient parallelism and improve cache utilization, loop_unroll and loop_reorder loops to group statements with similar addresses; improves code quality, reduces register pressure, and facilitates SIMD.
    6.“SIMD-ize” the code. The x86 compilers didn't generate good SSE code, so made a code generator to produce SSE intrinsics.
    7.Compress Data Structures (SpMV only). Since bandwidth limits performance, use smaller data structures: 16-bit vs. 32-bit index and smaller representations of non-zero subblocks
  6.3.1 Sparse Matrix-Vector Multiplication
    conventional implementations often run at less than **10%** of peak floating-point performance in uniprocessors. One reason is the irregular accesses to memory, which you might expect from sparse matrices
  6.3.2 Lattice-Boltzmann Magnetohydrodynamics
  6.3.3 Stencil
  6.3.4 3-D FFT
  6.3.5 Productivity vs. Performance
  6.3.6 Summary of Roofline Model Demonstration

# 7.FALLACIES ABOUT ROOFLINE
Fallacy: Doubling cache size will increase operational intensity
  Increasing cache size helps only with capacity misses and possibly conflict misses
Fallacy: The model doesn’t account for the long memory latency
Fallacy: The model ignores integer units in floating-point programs, which can limit performance
Fallacy: The model has nothing to do with multicore
  really push the limits of the memory system, considerable concurrency is necessary
Fallacy: You need to recalculate the model for every kernel
Fallacy: The model is limited to easily optimized kernels that never hit in the cache
Fallacy: The model is limited to floating-point programs

# 8.CONCLUSIONS


# Appendix A
Appendix A is found online at the CACM website: cacm.acm.org
https://cacm.acm.org/research-highlights/neural-architecture-search-as-program-transformation-exploration/
