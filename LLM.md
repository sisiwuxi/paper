# reference
- https://github.com/openppl-public

# title
- LLM inferefence
# Author
- sensetime HPC

# LLM
## infrastructure
- layernorm
- attention
- silu, activation
- matmul
- rotary embedding
- KV cache
## transformer_block
- series connection
- LLaMa 7B: 40*transformer_block
- flow
  - fp16 input
  - skip RMSNorm A & Quatize
  - int8
  - matmul for Q(int8), matmul for K(int8), matmul for V(int8)
    - mapping input to 3 differenct space
  - self attention fp16
    - concate three in one
  - matmul for O
    - linear mapping
  - skip RMSNorm B & Quatize
  - feedforward matmul A(int8)
  - mul & silu & dequantize & quantize
  - feedforward matmul B(int8)
- LLaMa 7B, batchsize=1, length=128
  - opname                  type          GPU_time
  - matmul for Q            GEMM          30
  - matmul for K            GEMM          31
  - matmul for V            GEMM          30
  - matmul for O            GEMM          31
  - skip RMSNorm A          Normalization 5 
  - feedforward matmul A    GEMM          70
  - feedforward matmul B    GEMM          71
  - feedforward matmul C    GEMM          73
  - skip RMSNorm B          Normalization 5 
  - attention               attention     9(Varies with length)

# LLM inference
## prefill
- generate KV cache for 1 user
- KV cache
  - S1
    - word embedding matrix
    - length = sequence length + hidden dimension(4096)
    - input = all token from user, any length
  - S2
    - matmul Q
    - matmul K
    - matmul V
  - S3
    - RQ = rotary embedding Q
    - RK = rotary embedding K
  - S4
    - KV cache = RK,V
- flash_attention = 2*matmul + softmax
  - S1
    - matrix Q = RQ
  - S2
    - TK = transpose K
    - matrix TK
  - S3
    - matrix QK, intermediate result
    - softmax per row
  - S4
    - matrix QK and V
## decoding 
- decode one by one token
- KV cache
  - S1
    - length = 1 + hidden dimension(4096)
    - input = last generated token
  - S2
    - matmul Q
    - matmul K
    - matmul V
  - S3
    - RQ = rotary embedding Q
    - RK = rotary embedding K
  - S4
    - KV cache += RK,V
    - history KV cache + new KV
- decoding_attention = 2*matmul + softmax
  - S1
    - matrix Q = RQ
  - S2
    - TK = transpose K
    - matrix TK
  - S3
    - matrix QK, intermediate result
    - softmax per row
  - S4
    - matrix QK and V

# LLM benchmark
## major factor
- thoughput
  - how many decoding per second
  - length dependency
  - fixed length, different model
- first token latency
  - prefill
  - < 2s
  - length dependency
- latency
  - 1 token decoding time
  - length dependency
- QPS
  - schedule full tasks on GPU
  - CNN: large batch
  - real input length, different length from different user
  - question per second = K_token/total_time

## LLaMa           fp16  H800  PPLNN.LLM   Cuda12.0
- model size(B)   gpu batch input_len output_len  prefill(ms) decode(ms)  throughput(token/s) mem(GB)
- 7               1   1     8         256         6.78        6.83        145.85              13.26
- 7               1   2     8         256         6.84        6.91        288.32              13.34
- 7               1   4     8         256         7.12        6.95        573.66              13.50
- 7               1   8     8         256         7.10        7.05        1131.10             13.82
- 7               1   16    8         256         7.94        7.15        2229.65             14.47
- 7               1   32    8         256         10.41       7.98        3992.18             15.77
- 7               1   64    8         256         16.35       8.55        7429.85             18.36
- 7               1   128   8         256         30.03       10.47       12095.66            23.57
- 7               1   256   8         256         58.60       15.20       16597.23            33.97
- 7               1   384   8         256         83.42       22.90       16536.86            44.36
- 7               1   512   8         256         110.82      25.45       19785.23            54.78
- 7               1   768   8         256         165.19      37.19       20298.52            75.57
- 7               1   1     1024      1024        30.68       8.64        115.40              13.86
- 7               1   2     1024      1024        58.33       8.75        227.09              14.56
- 7               1   4     1024      1024        109.13      8.88        445.36              15.93
- 7               1   8     1024      1024        213.48      9.23        848.04              18.70
- 7               1   16    1024      1024        415.95      10.06       1528.73             24.22
- 7               1   32    1024      1024        830.35      13.84       2184.17             35.27
- 7               1   64    1024      1024        1633.89     20.18       2939.07             57.37
- 7               1   80    1024      1024        2078.37     22.15       3309.25             68.42
  - batch = the number of upper limit = Up to 80 users at the same time

# PPL serving
- P:prefill
- D: decoding
-         first token latency   latency
- user1   PPPPPPPPPPPPPPPPPPPPPPDDDDDDDDDDDDDDDDDD...
- user2            PPPPPPPPPPPPPPPPPPPPPPPPDDDDDDDDDD...
- user3   PPPPPPPPDDDDDDDDDDDDDDDDDDDDD...
- user4                 PPPPPPPPDDDDDDDDDDDDDDDDDDDDD...

- dataset
  - https://github.com/openppl-public/ppl.llm.serving/blob/master/tools/samples_1024.json


# LLV inference
## sub-process
- tokenize: txt to vector
- computing: model inference
- detokenize: vector to txt
- sampling: sample based on inference reulst, select the most suitable word
  - beam serach
  - sample topK
  - argmax
  - fast_samp: https://github.com/openppl-public/ppl.llm.kernel.cuda/blob/master/src/ppl/kernel/llm/cuda/pmx/sample.cu
- return: immediately put like streaming
## prefill
- |tokenize|compute|fast_sample|return|
- tokenize 10%
- computing 80%
- sampling 10%
- return
## decoding
- |compute|fast_sample|detokenize|return|
- computing 80%
- sampling 10%
- detokenize < 5%
- return


# optimiza
## parallel heterogeneous device
- GPU: compute
- CPU: tokenize, detokenize
- parallel GPU and CPU
  - |---------prefilling----------------|
  - |tokenize|compute|fast_sample|return|
  -                  |compute|fast_sample|detokenize|return|
  -                          |compute|fast_sample|detokenize|return|
  -                                  |compute|fast_sample|detokenize|return|
  -                  |----------------------decoding-----------------------|
- QPS: speedup 10~20%

## dynamic batch schedule
- rearrange schedule user task
- GPU conflict
  - user1   |prefill|decoding|decoding|decoding|decoding|
  - user2           |prefill|decoding|decoding|decoding|decoding|
- merge user1 decoding and user2 prefill
  - user1   |prefill|merge|decoding|decoding|decoding|
  - user2           |merge|decoding|decoding|decoding|decoding|

## decoding_attention
- user1_decode_input[1, 4096]
  - |RMSNorm|matmul|attention|RMSNorm|matmul|
- user2_prefill_input[48, 4096]
  - |RMSNorm|matmul|attention|RMSNorm|matmul|
- merge_input[49, 4096]
  - |RMSNorm|matmul|flash_attention + decoding_attention|RMSNorm|matmul|
  - add mark to separate user1 and user2
  - matmul and normalize_on 4096 is indepent with user index
  - attention should use private KV cache, so we need split Q, KV cache
  - flash_attention for prefill
  - decoding_attention for decoding, flash_attention is twice as fast than flash_attention
- algorithm
  - https://github.com/openppl-public/ppl.llm.kernel.cuda/blob/master/src/ppl/kernel/llm/cuda/pmx/multi_head_cache_attention.cu
  - input 1 token, Q=1, KV cache is too large
  - based on cuda core
  - attn = query @ key.transpose(-2,-1)
  - attn = attn.softmax(-1)
  - return attn @ value

# VM allocator
## KV cache management
- speedup 200%
- private KV cache for each user
- variable length = input_length + decoding_length
- prefill generate input token cache with input_length
- dedocing generate 1 new output token and append after input token cache
## pytorch
- upper limit = allocate the largest length based on model
- LLaMa: 2048, 4096
- pros
  - fixed batch: QPS and batch maintain a linear relationship
    - before point: more batch -> more parallellism on GPU -> more throughput
      - 128, 12095
      - 256, 16597
      - 512, 19785
    - after point
      - 512 -> 512*3 = 1536, 2xxxxx
- cons
  - waste large memory size = low batch(#user)
## page attention
- 1 page frame = 8 token
- Linked lists implement links between page tables
- allocate 1 page frame for current user in Page fault exception
- pros
  - support more users, 2 or 3 times than pytorch
    - 512 -> 512*3 = 1536, 2xxxxx
- cons
  - freqently kill user
    - can not estimate the upper limit of user number
    - random kill user when OOM, and accept new user?
  - software implement, cut down 10~20% performance
  - batch = 768, do not need page attention
## VM allocator
- allocate predict length for user, satisfy 90% requirement
- kill user when memory is not enough
- pros
  - balance QPS and batch(#user)
## kill user
- swap to CPU is not recommended
  - flexgen
  - swap out, swap in
  - root cause: low bandwidth; 1G need 20ms
- baddwidth
  - GPU mem: 2,039GB/s
  - NVLink: 600GB/s
  - PCIe Gen4: 64GB/s

# KV cache quantization
## two parts need quantize
- quantize KV cache for increase #user
- quantize weight for decrease lantency
-       KV  weight  #user
- 0,0   N   N 
- 0,1   N   Q
- 1,0   Q   N       increase 100%
- 1,1   Q   Q
- model 7B, on 80GB
  - KV cache 84%, 67G, determined #user
  - weight 16%, 13G
- model 176B, on 8x80GB
  - KV cache 50%
  - weight 50%, 210G, 30G per card
## group quantization
- int8/int4 quantization
- insert Q/DQ operator
- fuse matmul and Q/DQ
- pytorch
  ```
    import torch
    
    # scale from whole group, 8 elements, high precision
    def group_quantize(x:torch.Tensor, group_size:int=8):
      x = x.reshape((-1, group_size))
      scale = torch.max(x.abs(), dim=-1, keepdim=True)[0] / 127
      qt = torch.round(x / scale).char()
      return qt.flatten().scale

    # scale from whole x, 8*N elements
    def per_tenosr_quantize(xL torch.Tensor):
      x = x.flatten()
      scale = torch.max(x.abs()) / 127
      qt = torch.round(x / scale).char()
      return qt.flatten().scale
  ```

# matmul quantization
## parser
- operator        GPU time percentage
- attention       7~30%
- matmul          70~90%
- normalization   3%
- other           <5%
- PPL.LLM use dynamic channel-token int8 quantize
- high precision, speedup 100% performance in large batch case

## quantization
- weight
  - preprocess during reading model
- activation: per channel-token dynamic int8 quantize
  - insert quantize operator before matmul_int8
  - insert dequantize operator after matmul_int8
  - fuse quantize in previous operator
  - fuse dequantize in latter operator
- matmul for O do not quantize
  - this matmul is not hot point
  - confused with fuse dequantize from previous later
  - input from attention, attention wil do softmax, the result of softmax do not conform the normal distribution

# int4 weight only vs int8
## int4 weight only 
- memory bound: phone
- end
  - batch = 1
  - normal
    - |load weight(13GB fp16) 90%|load input fp16|compute fp16|return fp16|
  - int4 quantize weight only
    - |load weight(3GB int8) 20%|dequantize int8 to fp16 10%|load input fp16|compute fp16|return fp16|
- service
  - batch = 128
  - normal
    - |load weight(13GB fp16) 30%|load input fp16|compute fp16|return fp16|
  - int4 quantize weight only
    - |load weight(3GB int8) 10%|dequantize int8 to fp16 30%|load input fp16|compute fp16|return fp16|

## int8 quantize
- compute bound
- quantize activate, weight and compute, both speedup
- service
  - batch = 128
  - normal
    - |load weight(13GB fp16) 30%|load input fp16|compute fp16|return fp16|
  - int8 quantize
    - |load weight(3GB int8) 10%|load input int8|compute int8|return int8|
- remove
  - dequantize int8 to fp16
- compute speedup
  - compute int8 quickly than compute fp16
- https://github.com/openppl-public/ppl.llm.kernel.cuda/pull/6
  ```
    def dequantize(...): pass

    def GEMM(X,W,output):
      for m in range(M):
        for n in range(N):
          for k in range(K):
            output[m,n] += X(m,k) * dequantize(W[k,n])
  ```

# FP8 vs INT8
- precision: int8 > fp8
- fp8 affinity on H800
- service: FP8~INT8 > INT4
- https://zhuanlan.zhihu.com/p/574825662


# INT4 vs nonlinear quantization
- customize quantization, linear quantization is not the best one
- https://github.com/TimDettmers/bitsandbytes
  - ./csrc/kernels.cu:169
  ```
  __device__ half dhDequantizeNF4(unsigned char val)
  {
    // the values for this tree was generated by test_normal_map_tree
    // in the file tests/test_functional.py
    if((val & 0b1000) == 8)
      if((val & 0b0100) == 4) // 1
        if((val & 0b0010) == 2) // 11
          if((val & 0b0001) == 1) // 111
            return 1.0f; 
          else
            return 0.7229568362236023f;
        else
          if((val & 0b0001) == 1) // 110
            return 0.5626170039176941f; 
          else
            return 0.44070982933044434f; 
      else
        if((val & 0b0010) == 2) //10
          if((val & 0b0001) == 1) // 101
            return 0.33791524171829224f; 
          else
            return 0.24611230194568634f; 
        else 
          if((val & 0b0001) == 1) // 100
            return 0.16093020141124725f; 
          else
            return 0.07958029955625534f; 

    else
      if((val & 0b0100) == 4) // 0
        if((val & 0b0010) == 2) //01
          if((val & 0b0001) == 1) // 011
            return 0.0f; 
          else
            return -0.09105003625154495f; 
        else
          if((val & 0b0001) == 1) // 010
            return -0.18477343022823334f; 
          else
            return -0.28444138169288635f;
      else
        if((val & 0b0010) == 2) //00
          if((val & 0b0001) == 1) // 001
            return -0.39491748809814453f;
          else
            return -0.5250730514526367f; 
        else 
          if((val & 0b0001) == 1) // 000
            return -0.6961928009986877f; 
          else
            return -1.0f; 

  }
  ```
# hardware selection
- https://pan.baidu.com/s/1hKlrHYeOBviQUopY3hUJ1g?pwd=3mv8
- predict performace on different hardware
- version               size  bandwidth power  max_request  throughput  compute_latency    access_latency       price Cost-effectiveness
-                       GB    GB/s      TFLOPS              tokens/sec  ms(60%)             ms(80%)
- TeslaV100-PCIe-16G    16    900       112     16          720         3.3                 22.22               38000 0.39
- TeslaV100-SXM2-32G    32    900       125     80          1800        14.6                44.44               47000 0.79
- RTX 400               24    1008      165.2   48          1612.8      6.6                 29.76               17500 1.89
- L40                   48    864       181.05  14          2073.6      18.1                69.44               58000 0.73
- L4                    24    300       121     48          480         9                   100                 19500 0.5
- A100-80G-SXM4         80    2039      312     272         5546        19.9                49.04
- A800-80G-SXM4         80    2039      312     272         5546        19.9                49.04               103000 1.1
- A100-80G-PCIe         80    1935      312     272         5263.1      19.9                51.68               140000 0.77 
- A800-80G-PCIe         80    1935      312     272         5263.2      19.9                51.68               9200   1.17
- A30                   24    933       165     48          5263.2      6.6                 32.15               28500  1.07
- A40                   48    696       149.7   144         1492.8      21.9                85.21               29500  1.16
- A10                   24    600.2     125     48          960.3       8.8                 49.98               16000  1.23
- A2                    16    200       18      16          160         20.3                100                 9500   0.35

- max_request = (size-weight)/KV_size
  - average_length = 256
- access_latency = size/bandwidth
- latency = max(compute_latency, access_latency)
- throughput = max_request*(1/latency)

- 176B
  - distribute
  - communication cost: NVLink