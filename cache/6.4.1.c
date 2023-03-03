// A.3 Measure Cache Line Sharing Overhead
// This section contains the test program to measure the overhead of using variables on the same cache line versus
// variables on separate cache lines.
// gcc 6.4.1.c -lpthread
#include <error.h>
#include <stdio.h>
#define __USE_GNU
#include <sched.h>
#include <pthread.h>
#include <stdlib.h>

#define N (atomic ? 10000000 : 500000000)
static int atomic;
static unsigned nthreads;
static unsigned disp;
static long **reads;
static pthread_barrier_t b;

static void *tf(void *arg)
{
  long *p = arg;
  if (atomic) {
    for (int n = 0; n < N; ++n) {
      __sync_add_and_fetch(p, 1);
    }
  } else {
    for (int n = 0; n < N; ++n) {
      *p += 1;
      asm volatile("" : : "m" (*p));
    }
  }
  return NULL;
}

int main(int argc, char *argv[])
{
  if (argc < 2)
    disp = 0;
  else
    disp = atol(argv[1]);
  if (argc < 3)
    nthreads = 2;
  else
    nthreads = atol(argv[2]) ?: 1;
  if (argc < 4)
    atomic = 1;
  else
    atomic = atol(argv[3]);
  pthread_barrier_init(&b, NULL, nthreads);
  void *p;
  posix_memalign(&p, 64, (nthreads * disp ?: 1) * sizeof(long));
  long *mem = p;

  pthread_t th[nthreads];
  pthread_attr_t a;
  pthread_attr_init(&a);
  cpu_set_t c;
  for (unsigned i = 1; i < nthreads; ++i) {
    CPU_ZERO(&c);
    CPU_SET(i, &c);
    pthread_attr_setaffinity_np(&a, sizeof(c), &c);
    mem[i * disp] = 0;
    pthread_create(&th[i], &a, tf, &mem[i * disp]);
  }
  CPU_ZERO(&c);
  CPU_SET(0, &c);
  pthread_setaffinity_np(pthread_self(), sizeof(c), &c);
  mem[0] = 0;
  tf(&mem[0]);
  if ((disp == 0 && mem[0] != nthreads * N) || (disp != 0 && mem[0] != N)) {
    error(1,0,"mem[0] wrong: %ld instead of %d", mem[0], disp == 0 ? nthreads * N : N);
  }
  for (unsigned i = 1; i < nthreads; ++i) {
    pthread_join(th[i], NULL);
    if (disp != 0 && mem[i * disp] != N) {
      error(1,0,"mem[%u] wrong: %ld instead of %d", i, mem[i * disp], N);
    }
  }

  return 0;
}
