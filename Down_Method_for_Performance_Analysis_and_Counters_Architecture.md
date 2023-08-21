# title
- A Top-Down Method for Performance Analysis and Counters Architecture

# Author
- Ahmad Yasin

# Institution
- Intel Corporation, Architecture Group

# Submitted
- Conference Paper · March 2014

# tags

# Abstract
- Optimizing an application’s performance for a given microarchitecture has become painfully difficult. Increasing microarchitecture complexity, workload diversity, and the unmanageable volume of data produced by performance tools increase the optimization challenges. At the same time resource and time constraints get tougher with recently emerged segments. This further calls for accurate and prompt analysis methods.
- In this paper a Top-Down Analysis is developed – a practical method to quickly identify true bottlenecks in out-of-order processors. The developed method uses designated performance counters in a structured hierarchical approach to quickly and, more importantly, correctly identify dominant performance bottlenecks. The developed method is adopted by multiple in-production tools including VTune. Feedback from VTune average users suggests that the analysis is made easier thanks to the simplified hierarchy which avoids the high learning curve associated with microarchitecture details. Characterization results of this method are reported for the SPEC CPU2006 benchmarks as well as key enterprise workloads. Field case studies where the method guides software optimization are included, in addition to architectural exploration study for most recent generations of Intel Core™ products.
- The insights from this method guide a proposal for a novel performance counters architecture that can determine the true bottlenecks of a general out-of-order processor. Unlike other approaches, our analysis method is low-cost and already featured in in-production systems – it requires just eightsimple new performance events to be added to a traditional PMU. It is comprehensive – no restriction to predefined set of performance issues. It accounts for granular bottlenecks in super-scalar cores, missed by earlier approaches.


# Problem
- Optimizing is difficult
  - microarchitecture complexity
  - workload diversity
  - huge volume of data
# Related work

# Key contribution
- Top-Down Analysis to quickly identify true bottlenecks in out-of-order processors

# Strengths
- low-cost
  - eight simple new performance events to be added to a traditional PMU
- already featured in in-production systems

# Weaknesses

# Contents

# Value
- V = innovation * effectiveness * scope
  - IF(impact factor)
  - JCR(journal citation reports)

# Key insights