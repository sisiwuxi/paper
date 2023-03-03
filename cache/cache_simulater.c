#include <stdio.h>
#include <unistd.h>
#include "time.h"
/*
cache line entries: chained in a circular list
*/
#define NPAD 7

struct l {
  struct l *n; // next cache line entry
  long int pad[NPAD]; // payload
};

int get_cache_line_size(void)
{
  FILE *fp = NULL;
  fp = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
  char buf[10];
  if(fp != NULL)
  {
    fgets(buf, 10, (FILE*)fp);
    printf("%s\n", buf);
  }
  fclose(fp);
  long l1_dcache_line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
  long l1_icache_line_size = sysconf(_SC_LEVEL1_ICACHE_LINESIZE);
  long l2_cache_line_size = sysconf(_SC_LEVEL2_CACHE_LINESIZE);
  long l3_cache_line_size = sysconf(_SC_LEVEL3_CACHE_LINESIZE);
  long l4_cache_line_size = sysconf(_SC_LEVEL4_CACHE_LINESIZE);
  printf("L1 dcache & icache line size is %ld, %ld bytes\n", l1_dcache_line_size, l1_icache_line_size);
  printf("L2 cache line size is %ld bytes\n", l2_cache_line_size);
  printf("L3 cache line size is %ld bytes\n", l3_cache_line_size);
  printf("L4 cache line size is %ld bytes\n", l4_cache_line_size);

  return(0);
}

int sum_row(void)
{
  const int row = 1024;
  const int col = 1024;
  int matrix[row][col];

  int sum_row = 0;
  for(int r=0; r<row; r++) {
    for(int c=0; c<col; c++) {
      sum_row += matrix[r][c];
    }
  }
  return sum_row;
}

int sum_col(void)
{
  const int row = 1024;
  const int col = 1024;
  int matrix[row][col];

  int sum_col = 0;
  for(int c=0; c<col; c++) {
    for(int r=0; r<row; r++) {  
      sum_col += matrix[r][c];
    }
  }

  return sum_col;
}

int add_2(void)
{
  int n = 1000l;
  int a[n];
  for (int i=0; i<=1000; i++) {
    a[i] = a[i] + 2;
  }

  return 0;
}

int add_next(void)
{
  int n = 1000l;
  int a[n];
  for (int i=0; i<=1000; i++) {
    a[i] = a[i] + a[i+1];
  }

  return 0;
}

int main(void)
{
  clock_t start, end;
  double duration;
  start = clock();
  end = clock();

  // get_cache_line_size();
  // sum_row();
  // sum_col();

  start = clock();
  add_2();
  end = clock();
  duration = (double)(end - start)/CLOCKS_PER_SEC;
  printf("add_2 duration=%f\n", duration);

  start = clock();
  add_next();
  end = clock();
  duration = (double)(end - start)/CLOCKS_PER_SEC;
  printf("add_next duration=%f\n", duration);  
  return(0);
}