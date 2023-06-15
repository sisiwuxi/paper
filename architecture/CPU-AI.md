# title
- Pushing the Limits of Machine Design: Automated CPU Design with AI

# Author
- Shuyao Cheng, Pengwei Jin, Qi Guo, Zidong Du, Rui Zhang, Yunhao Tian, Xing Hu, Yongwei Zhao, Yifan Hao, Xiangtao Guan, Husheng Han, Zhengyue Zhao, Ximing Liu, Ling Li, Xishan Zhang, Yuejie Chu, Weilong Mao, Tianshi Chen, Yunji Chen

# Institution
- State Key Lab of Processors, Institute of Computing Technology, Chinese Academy of Sciences
- University of Chinese Academy of Sciences
- Cambricon Technologies Corporation Limited
- University of Science and Technology of China
- Institute of Software, Chinese Academy of Sciences

# Submitted
- Wed, 21 Jun 2023 05:50:33 UTC

# tags
- Binary Speculation Diagram (BSD)
- Monte Carlobased expansion
- the distance of Boolean functions

# Abstract
- Design activity -- constructing an artifact description satisfying given goals and constraints -- distinguishes humanity from other animals and traditional machines, and endowing machines with design abilities at the human level or beyond has been a long-term pursuit. Though machines have already demonstrated their abilities in designing new materials, proteins, and computer programs with advanced artificial intelligence (AI) techniques, the search space for designing such objects is relatively small, and thus, "Can machines design like humans?" remains an open question. To explore the boundary of machine design, here we present a new AI approach to automatically design a central processing unit (CPU), the brain of a computer, and one of the world's most intricate devices humanity have ever designed. This approach generates the circuit logic, which is represented by a graph structure called Binary Speculation Diagram (BSD), of the CPU design from only external input-output observations instead of formal program code. During the generation of BSD, Monte Carlo-based expansion and the distance of Boolean functions are used to guarantee accuracy and efficiency, respectively. By efficiently exploring a search space of unprecedented size 10^{10^{540}}, which is the largest one of all machine-designed objects to our best knowledge, and thus pushing the limits of machine design, our approach generates an industrial-scale RISC-V CPU within only 5 hours. The taped-out CPU successfully runs the Linux operating system and performs comparably against the human-designed Intel 80486SX CPU. In addition to learning the world's first CPU only from input-output observations, which may reform the semiconductor industry by significantly reducing the design cycle, our approach even autonomously discovers human knowledge of the von Neumann architecture.

# Problem
- existing AI techniques only capable of generating correct circuit logic at most about 200 logic gates

# Related work
- artificial general intelligence (AGI)
- Nvidia’s adder tree 
- Google used the AI method to design chip floorplans 

# Key contribution
- We introduce the detailed methodology of the automated CPU design

# Strengths
- friendly: from only external input-output examples
- currect: generates large-scale Boolean function with almost 100% validation accuracy
- huge: search space of unprecedented size 10^(10^(540)) 
- quickly: generate an industrial-scale RISC-V CPU design within only 5 hours
- 65nm, 300 MHz; Linux (kernel 5.15) 

# Weaknesses
- need wide coverage of IO example
- Can not match the classic implementation or inefficient implementation

# Contents
## Learning the circuit logic of a CPU
- The traditional CPU design flow
```
             |--------------<-----------------|
  ISA -> LogicDesign -> CircuitLogic -> Verification -> CircuitLogic
   |________>____________ TestCases ____________| 
```
- The proposed flow to learn the CPU design from informal input-output examples
```
  ISA --> TestCases --> IO --> CircuitLearning --> CircuitLogic
```
- the target circuit logic
- y = xo OR x1
```
  x0--\__ y
  x1--/
```
- Binary Decision Diagram (BDD).
```
     x0
   0/  \1
  x1    x1
 0/\1  0/\1
 0  1  1  1
```
- Binary Speculation Diagram (BSD)
```
     x0
   0/  \1
   ()  ()
   0   1    
```
- The detailed learning process
  - Oracle -> F(x) -> F(x)=xiF(x|xi=0)+xiF(x|xi=1) -> BSD -> CircuitLogic
## Addressing accuracy and scalability challenges
- accuracy challenge: more input-output examples, increasing accuracy
- scalability challenge: similar nodes are clustered together for potential node merging
  - partition, expansion, and merging
## Evaluation
- IO 
  - 1789 input bits
  - 1826 output bits
  - IO examples is 1826 × 1798
- the layout of the entire chip
  - CPU core
  - Phase Locked Loop for clock generation (PLL)
  - Ethernet Media Access Controller module (EMAC)
  - SD controller (SDC)
- manufactured chip
  - ICT, CAS, 2021.12
- Dhrystone45 
- compare 
  - design: 5000h vs 5h
  - verification: 4560h 
## Discovering the von Neumann architecture
- Discovering the von Neumann architecture von Neumann architecture
- recursively decomposed into smaller functional modules
  - InputDevice
  - central processing unit
    - control unit
      - privilege controller
      - instruction decoder
    - arithmetic unit
      - ALU
        - arithmetic
      - CSR
      - LSU
        - address
      - other computation unit
  - memory unit
  - OutputDevice
## Conclusion
- the manual programming and verification process of traditional CPU design is completely eliminated

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights
- reforms the semiconductor industry
- building a self-evolving machine to beat the latest CPU designed by humanity eventually