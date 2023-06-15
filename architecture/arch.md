# architecture
- single-core: Intel
- multi-core: Intel
- many-core: Nvidia, Google
- huge-core: Cerebras
- heterogeneous: AMD
- ASIC: Application Specific Integrated Circuit

# compare
      memory        compute
GPU   RF,L1,L2      scalar, vector, tensorcore
TPU   RF,L1,L2      scalar, vector, SFU
DOJO  RF            scalar, vector, tensorcore
WSE   ASIC+SRAM

# AI generation
- V1
  - CPU/GPU <--> DDR
- V2
  - ACCELERATOR(TPU/NPU/ASIC) <--> DDR
- V3
  - ASIC+SRAM
  - Cerebras The Wafer-Scale Engine (WSE)

# Classification
## primary
- memory
- compute
## secondary
- number
  - transistors per core
  - cores per chip
- bandwidth
  - thread
  - warp
  - grid
  - chip
  - service
- area
- power
- frequency
- complexity of interconnection

# flow
- functionality
- correctness
- performance
- scalability
- hyperparameters

# software
- flexible
- easy control

---
---

# single core
## CPU frequency
## execution efficiency
- parallelism
  - intrinsic
    - hardware dependent
    - Instruction pipelining
    - multiple transmissions
    - out-of-order execution
  - data
    - SIMD: single instruction multiply data
    - vector program
      - vector length(bits)
        - 128,64,256,512
      - auto-vectorization
        - compiler option
      - Extension embedded interface
## SIMD Manufacturers

| Manufacturer | processor | ISA | vector length(bits) |
| ------ | ------ | ------ | ------ |
| motorola | G4 | AltiVec | 128 |
| DEC | Alpha | MVI | 64 |
| SGI | MIPS V | MDMX | 64 |
| Intel | Pentium | MMX | 64 |
| Intel | Pentium | SSE | 128 |
| Intel | Core | AVX | 256 |
| Intel | Core | IMCI | 512 |
| Sony | Cell | AltiVec | 128 |
| Sun | SPARC v9 | VIS | 64 |
| HP | PA-RISC | MAX-2 | 64 |
| AMD | Athlon | 3DNow! | 128 |
| ARM | ARMv6 | NEON | 128 |
| ARM | PPC970 | VMX | 128 |
| IBM | P6 | VMX | 128 |
| IBM | BG/L | / | 256 |
| LONGXIN | Godson | / | 256 |


---

# multi core
- multi processor integrated into one chip
- architecture
```
  CORE0 CORE1 CORE2 CORE3
   /|\   /|\   /|\   /|\
    |     |     |     |
   \|/   \|/   \|/   \|/
    -------------------                |- PCI-E 2.0 8x
    |       L3        |  <--> I/O <--> |- PCI-E 2.0 8x
    -------------------                |- JTAG
      /|\         /|\                  |- ethernet 
       |           |
      \|/         \|/         
      DMA0        DMA1
      /|\          /|\
       |            |
      \|/          \|/
      DDR3         DDR3
```

# many core
- XX, XXXX cores
- GPU
- High data density and simple logical branching
```
--------------------------------------------------------
|  stream_multiprocessor_1 ... stream_multiprocessor_n |
|  SP1 SP2 ... SPn             SP1 SP2 ... SPn         |
--------------------------------------------------------
                        /|\
                         |
                        \|/      PCI-E
                  device memory  <---->  Main_memory
                                            /|\
                                             |
                                            \|/
                                            CPU
```

# heterogenous
- Architecture follow AI model
- mix-programming
  - application rewrite
  - special optimization
```
  SPU   SPU        SPU
  L/S   L/S         LS
  /|\   /|\         /|\
   |     |    ...    |
  \|/   \|/         \|/ 
  MFC   MFC         MFC
  /|\   /|\         /|\
   |     |    ...    |
  \|/   \|/         \|/ 
   -------------------
   |      EIB        |
   -------------------
  /|\   /|\         /|\
   |     |    ...    |
  \|/   \|/         \|/ 
   L2
  /|\   /|\         /|\
   |     |    ...    |
  \|/   \|/         \|/     
  L1    MIC         BIC
  PPU   MainMemory  I/O device
```
- MFC: memory fabric controler
- MIC: memory controller
- BIC: External Bus IO interface
- EIB: Electrical Installation Bus
- LS: local storage

# ASIC
- Increasing the cost and efficiency of infrastructure
  - CPU inefficiency
  - GPU unsupport
- Google TPU SoC
```
Core        CPU     OtherProcessor
Memory
----------------------------------
|              BUS               |
----------------------------------
SoC_Interfaces  SoC_infrastructure     
```


---

# memory
## root cause
- un-balance between compute & memory
  - compute: 50%-100% per year
  - memory: 7% per year
## cycle
- compute
  - mul: 6
  - div: 20
- memory
  - 200 per LD/ST
- locality
- time
  - used data will be used later
- space
  - adjacent data has greater probability will be used later 
## memory wall
- solution: memory hierachy
- CPU <-> L1 <-> L2 <-> memory <-> IO
- expensive --> cheap
- quick --> slow
## compiler limitation
- DSL -> execution program
- main flow
  - serial/parallel high level program
  - C/C++ frontend
  - Interprocedural anlysis & optimization
  - loop recersive optimization
    - loop split
    - loop unblock
    - loop fission & fusion
    - loop verctorize automatically
      - Intel ICC -O3
  - global optimization
  - codegen
  - parallel objective program
- efficient
  - static analysis techniques
  - conservative rule for ensure correctness
  - tradeoff between generality & efficiency
  - time cost in optimizal stretagy
  - chanllege from quick upgrade and iteration of architecture and compute
## Programming Model
- CUDA， OpenACC, OpenCL, OpenMP, MPI