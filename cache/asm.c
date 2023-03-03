
void CSSETestDlg::ComputeArrayCPlusPlus(  
          float* pArray1,                   // [in] first source array  
          float* pArray2,                   // [in] second source array  
          float* pResult,                   // [out] result array  
          int nSize)                        // [in] size of all arrays  
{  
  
    int i;  
  
    float* pSource1 = pArray1;  
    float* pSource2 = pArray2;  
    float* pDest = pResult;  
  
    for ( i = 0; i < nSize; i++ )  
    {  
        *pDest = (float)sqrt((*pSource1) * (*pSource1) + (*pSource2)  
                 * (*pSource2)) + 0.5f;  
  
        pSource1++;  
        pSource2++;  
        pDest++;  
    }  
}  

void CSSETestDlg::ComputeArrayCPlusPlusSSE(  
          float* pArray1,                   // [in] first source array  
          float* pArray2,                   // [in] second source array  
          float* pResult,                   // [out] result array  
          int nSize)                        // [in] size of all arrays  
{  
    int nLoop = nSize/ 4;  
  
    __m128 m1, m2, m3, m4;  
  
    __m128* pSrc1 = (__m128*) pArray1;  
    __m128* pSrc2 = (__m128*) pArray2;  
    __m128* pDest = (__m128*) pResult;  
  
  
    __m128 m0_5 = _mm_set_ps1(0.5f);        // m0_5[0, 1, 2, 3] = 0.5  
  
    for ( int i = 0; i < nLoop; i++ )  
    {  
        m1 = _mm_mul_ps(*pSrc1, *pSrc1);        // m1 = *pSrc1 * *pSrc1  
        m2 = _mm_mul_ps(*pSrc2, *pSrc2);        // m2 = *pSrc2 * *pSrc2  
        m3 = _mm_add_ps(m1, m2);                // m3 = m1 + m2  
        m4 = _mm_sqrt_ps(m3);                   // m4 = sqrt(m3)  
        *pDest = _mm_add_ps(m4, m0_5);          // *pDest = m4 + 0.5  
          
        pSrc1++;  
        pSrc2++;  
        pDest++;  
    }  
}  


void CSSETestDlg::ComputeArrayAssemblySSE(  
          float* pArray1,                   // [输入] 源数组1  
          float* pArray2,                   // [输入] 源数组2  
          float* pResult,                   // [输出] 用来存放结果的数组  
          int nSize)                        // [输入] 数组的大小  
{  
    int nLoop = nSize/4;  
    float f = 0.5f;  
  
    _asm  
    {  
        movss   xmm2, f                         // xmm2[0] = 0.5  
        shufps  xmm2, xmm2, 0                   // xmm2[1, 2, 3] = xmm2[0]  
  
        mov         esi, pArray1                // 输入的源数组1的地址送往esi  
        mov         edx, pArray2                // 输入的源数组2的地址送往edx  
  
        mov         edi, pResult                // 输出结果数组的地址保存在edi  
        mov         ecx, nLoop                  //循环次数送往ecx  
  
start_loop:  
        movaps      xmm0, [esi]                 // xmm0 = [esi]  
        mulps       xmm0, xmm0                  // xmm0 = xmm0 * xmm0  
  
        movaps      xmm1, [edx]                 // xmm1 = [edx]  
        mulps       xmm1, xmm1                  // xmm1 = xmm1 * xmm1  
  
        addps       xmm0, xmm1                  // xmm0 = xmm0 + xmm1  
        sqrtps      xmm0, xmm0                  // xmm0 = sqrt(xmm0)  
  
        addps       xmm0, xmm2                  // xmm0 = xmm1 + xmm2  
  
        movaps      [edi], xmm0                 // [edi] = xmm0  
  
        add         esi, 16                     // esi += 16  
        add         edx, 16                     // edx += 16  
        add         edi, 16                     // edi += 16  
  
        dec         ecx                         // ecx--  
        jnz         start_loop                //如果不为0则转向start_loop  
    }  
}  
