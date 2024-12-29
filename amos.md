# Author
- Size Zheng
# Institution
- Peking University
# Submitted
- ISCA'22, June 18–22, 2022, New York, NY, USA
# Abstract
- Hardware specialization is a promising trend to sustain performance growth. Spatial hardware accelerators that employ specialized and hierarchical computation and memory resources have recently shown high performance gains for tensor applications such as deep learning, scientific computing, and data mining. To harness the power of these hardware accelerators, programmers have to use specialized instructions with certain hardware constraints.
- In this paper, we propose AMOS, which is an automatic compilation framework for spatial hardware accelerators. Central to this framework is the hardware abstraction that not only clearly specifies the behavior of spatial hardware instructions, but also formally defines the mapping problem from software to hardware. Based on the abstraction, we develop algorithms and performance models to explore various mappings automatically. Finally, we build a compilation framework that uses the hardware abstraction as compiler intermediate representation (IR), explores both compute mappings and memory mappings, and generates high-performance code for different hardware backends. Our experiments show that AMOS achieves more than 2.50× speedup to hand-optimized libraries on Tensor Core, 1.37× speedup to TVM on vector units of Intel CPU for AVX-512, and up to 25.04× speedup to AutoTVM on dot units of Mali GPU. The source code of AMOS is publicly available.
# Problem
- However, these hardware accelerators and instructions are quite new and there is a lack of understanding of the hardware abstraction, performance optimization space, and automatic methodologies to explore the space. Existing compilers use hand-tuned computation implementations and optimization templates, resulting in sub-optimal performance and heavy development costs. too many limitation to use tensorize on conv2d through primitives
# Related work
- Using Accelerators with Hand-optimized Libraries
  - CuDNNCuBlasCUTLASSHardware-aware Mapping
  - NowatzkiCoSASARAHASCOISA-aware Mapping
  - AutoTVMAnsorUNITXLAISA MapperTriamisuAKG
# Key insights
- mapping
  - feature [N,Hi,Wi,Ci]
  - weight  [R,S,Ci,Co]
  - output  [N,Ho,Wo,Co]
- basic mapping
  ```
    1
  N
  Ho  ->  M
  Wo
  Co  ->  N
  R
  S   ->  K
  Ci
  ```
- The number of possible complex mapping = 3^7 = 21087
- The number of valid mappings = (C(3,1)+C(3,2)+C(3,3)) * C(1,1) * (C(3,1)+C(3,2)+C(3,3)) = 7*1*7 = 49
  - C(3,1)+C(3,2)+C(3,3)=3+3+1=7
  - M
    - N
    - Ho
    - Wo
    - N,Ho
    - N,Wo
    - Ho,Wo
    - N,Ho,Wo
  - N
    - Co
  - K
    - R
    - S
    - Ci
    - R,S
    - R,Ci
    - S,Ci
    - R,S,Ci
- complex mapping
# Key contribution
- Mapping generation flow
- This example maps a small 2D convolution to Tensor Core (simplified to 2 × 2 × 2 multiplication).
  - Part a): Software description. 
  - Part b): software iterations for 2D convolution. 
  - Part c): intrinsic iterations for Tensor Core.
  - Part d) Iteration matching between software iterations and intrinsic iterations. 
  - Part e) and f): virtual compute and memory mapping without consideration for constraints of intrinsics. 
  - Part g) and h): physical compute and memory mapping with consideration for intrinsic constraints. 
  - Part i): virtual accelerator illustration. 
  - Part j): constraints in mapping.
# Detail
## INTRODUCTION
## BACKGROUND AND MOTIVATION
## 3. AMOS OVERVIEW
  - Hardware mapping: compute intrinsic
  - Constrain: fixed computation size & memory capacity
  - Memory mapping: memory intrinsic
  - Tuning & analytical model
## 4. HARDWARE ABSTRACTION
  - Compute mapping: sw_iter->intr_iter
  - Memory mapping: sw_iter->address(base+strides)
  - software iterations{n,ho,wo,co,kw,kh,ci}
  - intrinsic iterations{m,n,k}
## 5. MAPPING GENERATION, VALIDATION, AND EXPLORATION
  ```
    # image
        for n in range(N):
          for ho in range(Ho):
            for wo in range(Wo):
              for ci in range(Ci):
                for r in range(R):
                  for s in range(S):
                    fd_m = n*Ho*Wo + ho*Wo + wo
                    fd_k = ci*R*S + r*S + s
                    fdp[fd_m, fd_k] = f_pad_s[n,ci,ho+r,wo+s]
    # weight
        for co in range(Co):
          for ci in range(Ci):
            for r in range(R):
              for s in range(S):
                wd_n = co
                wd_k = ci*R*S + r*S + s
                wdp[wd_n, wd_k] = w_val[co,ci,r,s]
    # output
        for n in range(N):
          for ho in range(Ho):
            for wo in range(Wo):
              for co in range(Co):
                od_m = n*Ho*Wo + ho*Wo + wo
                od_n = co
                res[n,co,ho,wo] = odp[od_m,od_n]
  ```
## Tensor core compute intrinsic
  ```
  for m in range(M):
    for n in range(N):
      for k in range(K):
        Dst[m,n] += Src1[m,k] * Src2[n,k]
  ```
## performance model
- L = maximum(compute_latency, read_latency, store_latency)
- compute_latency can be estimated through Maestro and TENET
- read_latency = DataIn/in_bw
- store_latency = DataOut/out_bw

# 6. IMPLEMENTATION OF AMOS
## 7. EXPERIMENTAL RESULTS
- DNNs(Deep Neural Networks)
  - ShuffleNet
  - ResNet-18
  - ResNet-50
  - MobileNet-V1
  - BertMI-LSTM
- configurations of V100
  - the number of SM
  - the number of sub-core within one SM
  - memory size
  - evaluated bandwidth
- single operations
  - GEMV(GMV)
  - GEMM(GEM)
  - 1D Convolution(C1D)
  - 2D Convolution(C2D)
  - 3D Convolution(C3D): 3D -> 2D -> DOT
  - transposed 2D convolution(T2D)
  - group convolution(GRP)
  - dilated convolution(DIL)
  - depthwise convolution(DEP)
  - capsule convolution(CAP)
  - batched convolution(BCV)
  - grouped fully-connnected layer(GFC)
  - matrix mean and variance(MEM and VAR)
  - scan computation(SCN)
## 8. RELATED WORK

# Strengths
- give consideration to both hardware-aware and ISA-aware(tensorization)
  - #processor
  - Interconnect
  - Constrain
- explore various mappings automatically without templates
- enables systematic exploration of the mapping space
- allows flexible mapping for better performance

# Weaknesses
- Memory limitation
  - e.g.
    - wmma_m = wmma_n = wmma_k = 2
      - f[1,2,2] fp[1,4,4] fd[1,4,9] fdp[1,4,10]
        - 4 -> 16 -> 36 -> 40
      - w[4,3,3] wd[4,9] wdp[4,10]
        - 36 -> 36 -> 40
      - o[1,2,2,4] od[4,4]
        - 16 -> 16
  - e.g.
    - wmma_m = wmma_n = wmma_k = 16 
      - f[1,512,14,14] fp[1,512,16,16] fd[196,4608] fdp[208,4608]
        - 100352 -> 131072 -> 903168 -> 958464
      - w[512,512,3,3] wd[4608,512] wdp[4608,512]
        - 2359296 -> 2359296 -> 2359296
      - o[1,512,14,14] od[208,512]
        - 100352 -> 106496
- Pre-process kernel
- Post-process kernel

# Value
V = innovation * effectiveness * scope