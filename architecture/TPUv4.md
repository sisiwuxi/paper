
# title
- TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings

# Author
- Norman P. Jouppi, George Kurian, Sheng Li, Peter Ma, Rahul Nagarajan, Lifeng Nai, Nishant Patil, Suvinay Subramanian, Andy Swing, Brian Towles, Cliff Young, Xiang Zhou, Zongwei Zhou, and David Patterson

# Institution
- Google, Mountain View, CA
- berkeley

# Submitted
- 20230406
- ISCA '23, June 17–21, 2023, Orlando, FL, USA

# tags
- Computer systems organization → Architectures → Other architectures → Neural networks

# Abstract
- In response to innovations in machine learning (ML) models, production workloads changed radically and rapidly. TPU v4 is the
fifth Google domain specific architecture (DSA) and its third supercomputer for such ML models. Optical circuit switches
(OCSes) dynamically reconfigure its interconnect topology to improve scale, availability, utilization, modularity, deployment,
security, power, and performance; users can pick a twisted 3D torus topology if desired. Much cheaper, lower power, and faster
than Infiniband, OCSes and underlying optical components are <5% of system cost and <3% of system power. Each TPU v4
includes SparseCores, dataflow processors that accelerate models that rely on embeddings by 5x–7x yet use only 5% of die area and
power. Deployed since 2020, TPU v4 outperforms TPU v3 by 2.1x and improves performance/Watt by 2.7x. The TPU v4 supercomputer is 4x larger at 4096 chips and thus nearly 10x faster overall, which along with OCS flexibility and availability allows a large language model to train at an average of ~60% of peak FLOPS/second. For similar sized systems, it is ~4.3x–4.5x faster than the Graphcore IPU Bow and is 1.2x–1.7x faster and uses 1.3x–1.9x less power than the Nvidia A100. TPU v4s inside the energy-optimized warehouse scale computers of Google Cloud use ~2-6x less energy and produce ~20x less CO2e than contemporary DSAs in typical on-premise data centers.

# Problem
## how can we train ML models faster
- specialized hardware for ML
- distributed training on 100's or 1000's of accelerators
- flexible software for "interactive supercomputing"

# Related work
- 2019 TPUv3
- A100
- H100

# Key contribution
- BF16 = 1+8+7

# Strengths
- A100
  - performance x1.7
  - power x1.9
- > 1 exaflop/sec
- MLPerf
  - ResNet-50 14s, 29h on TPUv1
- distributed API: cross-compatible code
  - bit.ly/keras-TPU
  - detect haedware
  - find distribute strategy

# Weaknesses
- private model

# Contents
## 1 INTRODUCTION
## 2 RECONFIGURABLE OPTICAL SWITCH
- 2.1 Optical Circuit Switching(OCS)
- 2.2 Construction of the TPU v4 Supercomputer
- 2.3 OCS Availability Benefits
- 2.4 OCS Deployment Benefits
- 2.5 OCS Scheduling Benefits
- 2.6 OCS Modularity and Security Benefits
- 2.7 Tailoring OCS Topology to Improve Performance
- 2.8 Twisting the Torus
- 2.9 Distribution of Topologies
- 2.10 Cost of OCS Flexibility
## 3. SPARSECORE: EMBEDDINGS SUPPORT
- 3.1 Recommendation Models
- 3.2 Embeddings
- 3.3 Distributed Training
- 3.4 Key Performance Attributes
  - main: dense chip FLOPS/second
  - lookup: bottlenecked by the memory bandwidth, memory capacity, and VPU (vector processing unit) 
  - across chips: ICI(Inter-Core Interconnect) interconnection network
  - model parallelism: network bisection bandwidth
  - data parallelism: all-reduce operations injection bandwidth limits
  - reduce load imbalance
- 3.5 SparseCore
- 3.6 SparseCore Performance
## 4 USING ML TO TAILOR THE DNN TO THE TPU AND THE TPU TOPOLOGY TO THE DNN
## 5 PRODUCTION WORKLOAD PERFORMANCE
## 6 MLPERF BENCHMARK PERFORMANCE
## 7 DISCUSSION
- 7.1 Do peak FLOPS/second predict real performance?
- 7.2 How does OCS differ from NVLink and NVSwitch?
- 7.3 What if TPU v4 used IB versus OCS?
- 7.4 Nvidia announced the H100, the successor to A100, in 2022. Why not compare TPU v4 to it?
- 7.5 Why 30%–90% more power for A100 (Table 6)?
- 7.6 What is the CO2e from TPU v4 vs other DSAs?
- 7.7 How fast do ML workloads change?
- 7.8 Do individual DNN models also change?
- 7.9 Is MLPerf’s DLRM benchmark (Figure 14 above) realistic?
- 7.10 TPU v4 has less HBM capacity than A100; could that limit LLM performance?
- 7.11 How can DSAs avoid overspecialization?
## 8 RELATED WORK
## 9 SUMMARY


# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights
