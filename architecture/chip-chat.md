# title
- Chip-Chat: Challenges and Opportunities in Conversational Hardware Design

# reference
- https://arxiv.org/abs/2305.13243
- https://www.youtube.com/watch?v=6vC3t_soJok
- https://tinytapeout.com/runs/tt03/063/
- https://github.com/JBlocklove/tt03-qtchallenges-chatgpt_4/actions/runs/4803094867
- GitHub, “GitHub Copilot · Your AI pair programmer,” 2021. [Online].Available: https://copilot.github.com/
- “OpenLane,” May 2023, original-date: 2020-07-20T19:35:02Z. [Online].Available: https://github.com/The-OpenROAD-Project/OpenLane

# Author
- Jason Blocklove, Siddharth Garg, Ramesh Karri, Hammond Pearce

# Institution
- New York University
- University of New South Wales

# Submitted
- arXiv:2305.13243v1 [cs.LG] 22 May 2023

# tags
- Machine Learning (cs.LG); Hardware Architecture (cs.AR); Programming Languages (cs.PL)

# Abstract
- Modern hardware design starts with specifications provided in natural language. These are then translated by hardware engineers into appropriate Hardware Description Languages (HDLs) such as Verilog before synthesizing circuit elements Automating this translation could reduce sources of human error from the engineering process. But, it is only recently that artificial intelligence (AI) has demonstrated capabilities for machine-based end-to-end design translations Commercially-available instruction-tuned Large Language Models (LLMs) such as OpenAI's ChatGPT and Google's Bard claim to be able to produce code in a variety of programming languages; but studies examining them for hardware are still lacking. In this work, we thus explore the challenges faced and opportunities presented when leveraging these recent advances in LLMs for hardware design. Using a suite of 8 representative benchmarks, we examined the capabilities and limitations of the state of the art conversational LLMs when producing Verilog for functional and verification purposes. Given that the LLMs performed best when used interactively, we then performed a longer fully conversational case study where a hardware engineer co-designed a novel 8-bit accumulator-based microprocessor architecture. We sent the benchmarks and processor to tapeout in a Skywater 130nm shuttle, meaning that these 'Chip-Chats' resulted in what we believe to be the world's first wholly-AI-written HDL for tapeout.

# Problem
- Copilot suggest bad open source code of verilog

# Related work
- Dave: GPT2 from 2020
- GPT3.5
- GPT4
- Data Repository for Chip-Chat, Available: https://zenodo.org/record/7953724
- CPU
  - 8bit accumulator
  - CU: control unit
  - ALU: arithmetic logic unit
  - register
    - acc: accumulator
    - PC: program counter
    - IR: intrinsic register
  - BUS
    - control bus -> CU
    - address bus -> MAR
    - data bus -> MDR
- GPT
  - generative transformer
  - pre-trained on unlabled-data
  - finetune on labled-data
  - domain specific application
    - QA
    - translate
    - write
    - coding
    - summary
    - sentiment analysis
- HDL
  - hardware description language
- EDA
  - electronic design automation
- design
  - system  -> logical -> pyhiscal -> mask&manufacture -> test&package
  - system: contraint
  - logical: frontend
    - implement&development -> function simulation -> FPGA verification -> logic synthesis -> pre-function simulation
      - implement&development: output = **RTL soft IP** = *.v
      - logic synthesis: output = netlist film IP
  - physical: backend

# Key contribution
- initial specifications

# Strengths
## flow
- prompt -> LLM -> design.v -> synthesis + verification -> fab -> IC
- initial LLM prompt
  - inputs: clock, active-low reset, data 1bit, shift enable
  - outputs: data 8bits
- iVerilog simulation
- error feedback
  - tool
  - simple human
  - moderate human 
  - advanced human
## LLM
- 8*LLM
- pass
  - chatGPT3.5
  - chatGPT4
  - github
- fail
  - Bard
  - HuggingChat
## tapeout
- Skywater, shuttle, tapeout, 130nm
## IP
- ALU
- ACC
- IR
- PC
- mux
  - PC, ACC, Mem
## testbench

# Weaknesses
## verification
- the test coverage was generally very poor, how can you really trust these models
- self-checking test bench
- 10 times amount of design time on verification 
## prompt engineer
- professional feedback
- prompt base learning

# Contents

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights
- AI generate verilog
