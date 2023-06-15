# Evaluation indicators for program performance
- execution time
- computing efficiency & access efficiency
- throughput & delay
- speedup
- Calculate optimization upper limit
  - Amdahl
  - Gustafson

# execution time
- C
  - time
  - clock
- linux
  - gettimeofday
  - time ./a.out
    - real
    - user
    - sys
- asm
  - RPCC
  - code
    ```
      unsigned long rpcc() {
        unsigned long result;
        unsigned hi,lo;
        asm volatile("rdtsc":"=a"(lo),"=d"(hi));
        result = ((unsigned long long)lo)|(((unsigned long long)hi)<<32);
        return result;
      }
      int main() {
        unsigned long b[m], start, end, k;
        for (j=0;j<10;j++) {
          start = rpcc();
          fun(16);
          end = rpcc();
          b[j] = end - start;
          k += b[j];
        }
      }
    ```

# computing efficiency & access efficiency
## computing efficiency
- CE = Real / Theoretical floating point performance
- CPU > 50%
- GPU > 40%
## access efficiency
- AE = Effective / Theoretical bandwidth
- BW > 30%

# throughput & delay
- software system
## throughput
- workloads in unit time
- quantity
## delay
- time used from start to end
- quickly

# speedup
- SU = before/after

# Amdahl
- S: total speedup
- S = 1/((1-a) + a/n)
  - a: parallel partition percentagy
  - n: speedup from parallel partition
- example
  - a = 50%, n = 1.15
  - S = 1/((1-0.5) + 0.5/1.15) = 1/(0.5 + 0.43) = 1.08
  - seraial part = 50%, upper speedup = 1/0.5 = 2
- cons
  - The computing scale does not increase with the increase of processors
  
# Gustafson
- increase the number of processors

## expansion speedup
- S = n + (1-n)*f = f-n(f-1)
  - f: the number of processor
  - n: serial partition percentage
- the number of processor and the expansion sppedup are in the direct ratio
- serial partition percentage is not the bottleneck
- potential of parallel

## flow
- target of performance 
- profiling to get performance data
- find out the bottleneck through parse the data
- amend program for performance
- validation correctness and effectiveness
- end of current iteration
- Determine whether it meets the requirements

## example
- original
```
void matrixmulti(int N, int x[n][n], int y[n][n], int z[n][n]) {
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      r = 0;
      for (k=0; k<N; k++) {
        r = r + y[i][k] * z[k][j];
      }
      x[i][j] = r;
    }
  }
}
```
  - parse
    - name        percentegy time(s) total_time(s) times
    - matrixmulti 98.08%  3.07  3.13  1
    - main        1.92%   0.06  -     -

- bottleneck = load un-continous 
  - z[k][j]
  - solution = loop reorder, move k to the outmost
  ```
  void matrixmulti(int N, int x[n][n], int y[n][n], int z[n][n]) {
    for (k=0; k<N; k++) {
      for (i=0; i<N; i++) {
        r = y[i][k];
        for (j=0; j<N; j++) {
          x[i][j] += r*z[k][j];
        }
      }
    }
  }
  ```
  - parse
    - name        percentegy time(s) total_time(s) times
    - matrixmulti 97.84%  2.26  2.31  1
    - main        2.16%   0.05  -     -

- unroll
  ```
  void matrixmulti(int N, int x[n][n], int y[n][n], int z[n][n]) {
    for (k=0; k<N; k++) {
      for (i=0; i<N; i++) {
        r = y[i][k];
        for (j=0; j<N-4; j+=4) {
          x[i][j] += r*z[k][j];
          x[i][j+1] += r*z[k][j+1];
          x[i][j+2] += r*z[k][j+2];
          x[i][j+3] += r*z[k][j+3];
        }
      }
    }
  }
  ```
  - parse
    - name        percentegy time(s) total_time(s) times
    - matrixmulti 98.53%  2.01  2.04  1
    - main        1.47%   0.03  -     -

---

