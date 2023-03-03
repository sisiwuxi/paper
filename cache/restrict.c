#include <stdio.h>

/*
 * @brief 
 * 
 * @param a 
 * @param b 
 * @return int
 * 
 * 0000000000001180 <add1>:
    1180:       f3 0f 1e fa             endbr64 
    1184:       c7 07 0a 00 00 00       movl   $0xa,(%rdi)  ; *a=10
    118a:       c7 06 0c 00 00 00       movl   $0xc,(%rsi)  ; *b=12
    1190:       8b 07                   mov    (%rdi),%eax  ; res=10
    1192:       83 c0 0c                add    $0xc,%eax    ; res=10+12=22
    1195:       c3                      retq   
    1196:       66 2e 0f 1f 84 00 00    nopw   %cs:0x0(%rax,%rax,1)
    119d:       00 00 00 
 *
 */
int add1(int* a, int* b)
{
  *a = 10;
  *b = 12;
  return *a + *b;
}

/*
 * @brief 
 * 
 * @param a 
 * @param b 
 * @return int 
 * 00000000000011b0 <add2>:
    11b0:       f3 0f 1e fa             endbr64 
    11b4:       c7 07 0a 00 00 00       movl   $0xa,(%rdi)  ; *a=10
    11ba:       b8 16 00 00 00          mov    $0x16,%eax   ; res=22
    11bf:       c7 06 0c 00 00 00       movl   $0xc,(%rsi)  ; *b=12
    11c5:       c3                      retq   
    11c6:       66 2e 0f 1f 84 00 00    nopw   %cs:0x0(%rax,%rax,1)
    11cd:       00 00 00 
 */
int add2(int* __restrict a, int* __restrict b)
{
  *a = 10;
  *b = 12;
  return *a + *b;
}


/*
 * gcc restrict.c -O3
 * objdump -d a.out
*/
int main(int argc, char *argv[])
{
  int a = 0;
  int b = 0;
  int res_o3 = 0;
  int res_o3_restrict = 0;
  printf("\n ===== a + b ===== \n");
  res_o3 = add1(&a, &b);
  printf("\nres_o3=%d\n",res_o3);
  res_o3_restrict = add2(&a, &b);
  printf("\nres_o3_restrict=%d\n",res_o3_restrict);

  printf("\n ===== a + a ===== \n");
  res_o3 = add1(&a, &a);
  printf("\nres_o3=%d\n",res_o3);
  res_o3_restrict = add2(&a, &a);
  printf("\nres_o3_restrict=%d\n",res_o3_restrict);
  return 0;
}
