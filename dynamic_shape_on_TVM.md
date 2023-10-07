# title
- Author: zhuwenxi
- Institution: tencent AI lab
- Submitted: 20230916

# tags

# Abstract

# Problem

# Related work
- DLight based on relex

# Key contribution

# Strengths

# Weaknesses

# Contents
## AIGC HPC
- model
  - bert/GPT-1, GPT-2, GPT-3, LaMDA/Disco Diffusion, GPT-3.5/4 Stable Diffusion, hunyuan
  - https://aiarena.tencent.com/
- require
  - environment
  - data
  - algorithm
  - resource
## chanllege of dynamic shape optimization
- LLM
- classification
  - symbolic dimensions
    - rank is static
    - dimension: [x, 512], [x+128, 512]
  - symbolic rank
    - rank is dynamic
    - tensor[a,b,c] or tensor[a,b,c,d]
  - runtime dimensions
    - rank is static
    - dimension is determined in runtime
    - Unique
  - runtime rank
    - rank is dynamic
    - rank and dimension is determined in runtime
  - variable
    - dimension is different in tensor
    - ragged batch
- TVM limitation
  - auto schedule
    - designed for static shape
    - target
      - access locality
      - ILP: vector instruction, pipeline stream
      - TLP: match multi-core, multi-process
  - dynamic shape
    - program describe: use symbol instead of number
    - schedule algorithm: split, data rearrangement
  - cost model
    - sample partial shape and statistics
  - limitation
    - split
      - static shape
        - avoid remainder
        - prime number
          - finite factor
          - can not match high performance implement
      - dynamic shape
        - have to consider remainder
        - can not obtain specific number during compile stage
        - keypoint: tradeoff between split size and remainder size
    - large and small blocks
      - first level: high performance block size for major part
      - second level: low performance block size for remainder part
    - padding
      - pad zero to match the high performance block size
      - introducing additional computational complexity
    - all use high performance blocks

## TVM solution
- compare
  - traditional mathmatical library
    - manual rule to select HP kernel
    - major part use the selected kernel
    - remainder part use another small kernel
    - cons
      - can not cover all shapes
      - can not do sub-graph fusion
  - AI compiler
    - can generate all shape HP kernel
    - codegen support all type sub-graph fusion
- incorporate
  - matmul = combination of kernels
    - select suitable cache block HP kernel
    - determined kernel schedule = determined whole schedule
  - kernel = combination of micro-kernel
    - determined shape kernel = HP micro-kernel(register-block)
  - HP kernel
    - tvm codegen
    - select suitable combination with many micro-kernel
- algorithm
  - remainder
    - local padding in cache intead of global outside padding
  - data rearrange
    - pack/pre-pack
    - online pack is better than audo-scheduler
  - genrate kernel
    - N: multiply of register length
    - M
      - as more register as possible which satisfy the register limitation
      - pre-tuning to save HP register-block combination
    - multiply of shapes in micro-kernel set
  - kernel dispatch
    - solution 1
      - search
        - candidate set
        - remove low performance candidate based on padding size and kernel shape
        - simulate on hardware and select the best kernel and combination with micro-kernel
      - performance
        - better than mathmatical library and auto-scheduler
      - time cost
        - N minutes
    - solution 2
      - flow
        - use theoritical performance compute function
        - select kernel with the maximum shape using minimum padding size
        - obtain better balance between high cache locality and low padding cost
      - performance
        - comparable with solution 1, partial shape have 3% performance loss
      - no search time cost
  - DLight
    - dynamic-aware light-weight scheduler
    - support dynamic shape
    - do not need tuning or lowest search cost
    - based on rule instead of kernel template
      - one rule can cover a mount of operation
      - multiple rules can be freely combined
    - easy to customize
      - implement by python
      - high efficiency
    - based on relax
    - analyze-then-schedule
      - same feature between many operators
        - LLaMa decoding: GEMV in FFN and MHA
        - reduction-dominant: softmax, layernorm, RMSNorm
      - share same optimize in cache locality and parallelism with same operator
        - similar optimize schedule
        - tiling for GEMM
    - rule
      - reduction, flash attention, GEMV, GEMM/Conv
    - combination
      - a rule for GPU
      - a rule for CPU
- evaluation
  - M=N=K on CPU

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights