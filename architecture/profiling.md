# Analysis and Measurement of Program Performance
- HW
  - processor, memory, SRMA, network, BUS, NoC
  - Scalability, Parallellism, Capacity
- SW
  - algorithm
  - implementation
  - resource allocation
# profiling
- HW
- kernel
- system scheduler
- application
# Common performance issues
- large amount of resource not release
- frequent I/O operation
- cache miss
- program efficient
# Common performance method
- visit block
  - time
  - space
- wait list
  - end - ready
- little
  - L = lambda*W
  - L: average number of task
  - lambda: new task ratio
  - W: averate time of task
- parallel = latency * throughput
  
# measure
- data type
- tool type
- tool
## data type
- summary
  - processor utilization
  - average response time of I/O
- detail: time, line
- snapshot
## tool type
- system
- process
- principle
  - counter
    - unsigned int
    - vmstat: memory
      - vmstat -a 2 10
    - iostat: I/O
      - iostat -d 2 3
    - top: processor
    - ps: process
      - ps -ef
      - ps -aux|grep bash
  - tracker
    - gdb
      - break
      - p, whatis
      - dynamic change value
    - pstack
      - pstack PID
      - whatpid()
    - strace
      - strace ls -al
      - strace ./add.out
      - strace -p PID
  - monitor
    - nestat
      - nestat -a|head
    - sar
      - $ sar -u 1 3
    - iotop
    - mpstat
      - mpstat-P ALL 2 3
  - profile
    - gprof
      - GNU
      - $ gcc loop_permute.c -pg;./a.out; generate gmon.out
      - $ gprof a.out
    - oneAPI
      - intel
    - perf
      - $ gcc loop_permute.c -g
      - $ perf stat ./a.out
    - nvprof: CUDA
## profile
- Sampling and analyzing the state of the system at specific time intervals
- find out bottleneck, hotpoint
- flow
  - source code
  - insert profile code
  - compile & run
  - parse info

---

# system configuration
## processor
- more /proc/cpuinfo
- cat /proc/cpuinfo | grep "physical id"
- cat /proc/cpuinfo | grep "cpu cores"
- nice
  - renice -n 9 PID
  - chrt -p PID
  - chrt -p -f 10 PID
- taskset
  - taskset -pc 7-10 PID
- /proc/irq/number/smp_affinity
- numactl --membind 1 --cpunodebind 1 --localalloc APP
## memory
- cat /proc/meminfo
- dmidecode | grep -A16 "Memory Device"
- APP increase
  - stack
  - heap
  - text
- APP not increase
  - swap
  - shared
- EXEC_P_AGESIZE
  - /usr/src/linux-hwe-5.11-headers-5.11.0-27/arch/x86/include/asm/elf.h
  - echo 50 > /proc/sys/vm/nr_hugepages
  - grep Huge /proc/meminfo
- swap
  - swap_tendency = mapped_ratio/2 + distress + vm_swapiness
- KSM
  - kernel samepage merging
  - /etc/ksmtuned.conf
- ulimit
  - cgroup
  - memory.memsw.limit_in_bytes
  - memory.limit_in_bytes
  - memory.swappiness
  - memory.oom_control