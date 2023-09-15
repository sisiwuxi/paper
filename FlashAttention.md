# title
- Modest understanding on LLM
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

# Author
- Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

# Institution

# Submitted
- https://arxiv.org/abs/2205.14135
- https://tridao.me/publications/flash2/flash2.pdf
- https://github.com/Dao-AILab/flash-attention.git

# tags

# Abstract
- Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware -- accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3× speedup on GPT-2 (seq. length 1K), and 2.4× speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

# Problem
- Attention has time and memory complexity quadratic in sequence length O(n^2)
- most operations in Transformers are bottlenecked by memory accesses 
  - database joins
  - image processing
  - numerical linear algebra

# Related work
## reduce the compute and memory requirements of attention
- sparse-approximation
- low-rank approximation
- cons: focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO)
## IO-Aware Runtime Optimization
- Efficient ML Models with Structured Matrices
- Sparse Training
- Efficient Transformer
## starting point
- Apex’s FMHA code: https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha

# Key contribution
- IO-aware
  - avoid reading and writing the attention matrix to and from HBM
    - computing the softmax reduction without access to the whole input
    - not storing the large intermediate attention matrix for the backward pass
  - recomputation replace HBM access
- techniques
  - tiling: restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction
  - store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass
- IO complexity = 𝑂 (N^2 𝑑^2 𝑀^{−1})
  - d: head dimention
  - M: size of SRAM
- block-sparse FlashAttention

# Strengths
- widely adopted such as PyTorh2.0
- open-source， FlashAttention and FlashAttention-2 are free to use and modify

# Weaknesses
- in the context of model training, the intermediate values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion

# Contents

## 2 Background
### 2.1 Hardware Performance
- GPU Memory Hierarchy
  - smaller， faster
    - type        memory            bandwidth
    - CPU_DRAM    1TB               12.8GB/s
    - GPU_HBM     40-80GB           1.5-2.0TB/s 
    - GPU_SRAM    20MB=192KB*108SM  19TB/s
- Performance characteristics
  - arithmetic intensity: the number of arithmetic operations per byte of memory access
  - Compute-bound
    - matrix multiply with large inner dimension(big N)
    - convolution with large number of channels(Co)
  - Memory-bound
    - elementwise (e.g., activation, dropout)
    - reduction (e.g., sum, softmax, batch norm, layer norm)
    - most common approach: kernel fusion
### 2.2 Standard Attention Implementation
- N: the sequence length, 1024
- 𝑑: the head dimension, 64
- standard_attention
  - Algorithm_0
    - Input: x[N]
    - Q[N, d]
    - K[N, d], Kt[d, N]
    - S[N, N] = Q @ Kt
      - S[N, N] = 1/sqrt(dim) * S
      - S[N, N] = dropout(S)
    - P[N, N] = softmax(S), row-wise
    - P[N, N] = mask(P), Lower triangular matrix
    - V[N, d]
    - O[N, d] = P @ V
  - Algorithm 0 Standard Attention Implementation
    - Require: Matrices Q, K, V ∈ R N ×𝑑 in HBM.
    - 1: Load Q, K by blocks from HBM, compute S = QK> , write S to HBM.
    - 2: Read S from HBM, compute P = softmax(S), write P to HBM.
    - 3: Load P and V by blocks from HBM, compute O = PV, write O to HBM.
    - 4: Return O.
  - HBM = Q + K + S + P + V + O = Nxd + Nxd + Nxd + NxN + NxN + Nxd
  - memory-bound
    - small N matmul: 64<1024
    - softmax
- fused_kernel
  - O[N, d] = softmax_mask(Q @ Kt) @ V

## 3 FlashAttention: Algorithm, Analysis, and Extensions

### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation
- Tiling
  - decompose the large softmax with scaling
    - standard softmax
      - softmax(x) = exp(x) / sum(exp(x))
    - avoid overflow
      - m(x) = max(x)
      - f(x) = [ exp(x)/exp(m(x)) ]= [ exp(V-m(x)) ]
      - l(x) = sum(f(x)) = sum(exp(x-m(x)))
      - softmax(x) = f(x)/l(x)
    - concatenated softmax
      - m(x) = m(x1 x2) = max(m(x1) m(x2))
      - f(x) = [exp(m(x1) - m(x))*f(x1) exp(m(x2) - m(x))*f(x2)]
        - m(x1) == m(x), f(x1)_update = f(x1)
        - m(x1) != m(x), f(x1)_update = exp(x1)/exp(m(x)) = exp(x1)/[exp(m(x1)) * (exp(m(x))/exp(m(x1)))] = (exp(x1)/exp(m(x1)))*(exp(m(x1))/exp(m(x))) = f(x1) * exp(m(x1)-m(x))
      - l(x) = l(x1 x2) = sum(f(x1) f(x2)) = sum(exp(x1 - m(x))) + sum(exp(x2 - m(x))) = exp(m(x1)-m(x))*sum(exp(x1-m(x1))) + exp(m(x2)-m(x))*sum(exp(x2-m(x))) = exp(m(x1)-m(x))*l(x1) + exp(m(x2)-m(x))*l(x2)
      - softmax(x) = f(x)/l(x)
- Recomputation
  - backward
    - storing the output O and the softmax normalization statistics (𝑚, ℓ)
    - recompute the attention matrix S and P
  - reduced HBM accesses

- Algorithm 1 FlashAttention
  - Require: Matrices Q, K, V ∈ R[𝑁×𝑑] in HBM, SRAM of size 𝑀
    - 1: Set block sizes 𝐵𝑐 = 𝑀/4𝑑, 𝐵𝑟 = min(𝑀/4𝑑 , 𝑑)
    - 2: Initialize O = (0)[𝑁×𝑑], ∈ R[𝑁×𝑑], ℓ = (0)N ∈ R^N , 𝑚 = (−∞) N ∈ R N in HBM
    - 3: Divide Q into 𝑇𝑟 = N/Br blocks Q1 ,..., Q𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, and divide K, V in to 𝑇𝑐 = N/Bc blocks K1 ,..., K𝑇𝑐 and V1 ,..., V𝑇𝑐 , of size 𝐵𝑐 × 𝑑 each.
    - 4: Divide O into 𝑇𝑟 blocks O𝑖 ,..., O𝑇𝑟 of size 𝐵𝑟 × 𝑑 each, divide ℓ into 𝑇𝑟 blocks ℓ𝑖 ,..., ℓ𝑇𝑟 of size 𝐵𝑟 each, divide 𝑚 into 𝑇𝑟 blocks 𝑚1 ,..., 𝑚𝑇𝑟 of size 𝐵𝑟 each.
    - 5: for 1 ≤ 𝑗 ≤ 𝑇𝑐 do
    - 6:    Load K𝑗 , V𝑗 from HBM to on-chip SRAM
    - 7:    for 1 ≤ 𝑖 ≤ 𝑇𝑟 do
    - 8:      Load Q𝑖 , O𝑖 , ℓ𝑖 , 𝑚𝑖 from HBM to on-chip SRAM
    - 9:      On chip, compute S𝑖𝑗 = Q𝑖 K𝑇𝑗 ∈ R[𝐵𝑟×𝐵𝑐]
    - 10:     On chip, compute 𝑚˜𝑖𝑗 = rowmax(S𝑖𝑗) ∈ R𝐵𝑟 , P̃𝑖𝑗 = exp(S𝑖𝑗 − 𝑚˜𝑖𝑗)∈ R 𝐵𝑟×𝐵𝑐 (pointwise), ℓ˜𝑖𝑗 = rowsum( P̃𝑖𝑗 ) ∈ R𝐵𝑟
    - 11: On chip, compute 𝑚 𝑖new = max(𝑚𝑖, 𝑚˜𝑖𝑗) ∈ R𝐵𝑟 , ℓ𝑖new = 𝑒(𝑚𝑖 −𝑚𝑖 ℓ𝑖 + 𝑒 𝑚˜ 𝑖 𝑗 −𝑚𝑖 ℓ˜𝑖𝑗 ∈ R 𝐵𝑟 .
    - 12:     Write O𝑖 ← diag(ℓ𝑖new) −1 (diag(ℓ𝑖 )𝑒(𝑚𝑖−𝑚𝑖) O𝑖 + 𝑒(𝑚˜𝑖𝑗−𝑚𝑖)P̃𝑖𝑗V𝑗) to HBM.
    - 13:     Write ℓ𝑖 ← ℓ𝑖new , 𝑚 𝑖 ← 𝑚 𝑖new to HBM
    - 14:    end for
    - 15: end for
    - 16: Return O
### 3.2 Analysis: IO Complexity of FlashAttention
- Forward + backward
  - Fewer HBM accesses result in faster runtime
  - block-sparse FlashAttention is faster than FlashAttention
- Theorem 2
  - N = the sequence length, 𝑑 = the head dimension, and 𝑀 = size of SRAM with 𝑑 ≤ 𝑀 ≤ N
    - d: 64-128
    - M: 100KB
  - HBM accesses
  - Standard: 𝑁𝑑 + N^2
  - FlashAttention: N^2 d^2 M^(-1)
### 3.3 Extension: Block-Sparse FlashAttention
- Q[N, 𝑑], K[N, 𝑑], V[N, 𝑑] and a mask matrix M̃[N, N] ∈ {0, 1}
- S[N, N] = QKt
- P[N, N] = softmax(S . M̃ )
- O[N, 𝑑] = PV
- fixed butterfly sparsity pattern 
## 4 Experiments
### 4.1 Faster Models with FlashAttention
- Training Speed: fast
### 4.2 Better Models with Longer Sequences
- Language Modeling with Long Context
- Long Document Classification
- Path-X and Path-256
### 4.3 Benchmarking Attention
- Runtime
- Memory Footprint
## 5 Limitations and Future Directions
- Compiling to CUDA: PyTorch + Halide
- IO-Aware Deep Learning
- Multi-GPU IO-Aware Methods: IO analysis layer
## constrain
- S[n^2, n^2]，很大，SRAM:128KB放不下；
- 计算S/P/O会浪费大量时间在搬运数据上，比如P，需要把S一点点从HBMload进来，softmax后再搬出到HBM。

## optimize
- 算法优化：
  - mask_softmax_fuse
  - dense_scale_mask_softmax_dropout_dense_fuse
  - QKV split

## scale softmax
  - split 1 vector to 2 block
  - max(block)
  - scale previous softmax
  - Update sum(exp(x))
  - scale current softmax
  - Update P

## Triton

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights



