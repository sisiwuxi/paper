# # name
# VMCU: C OORDINATED M EMORY M ANAGEMENT AND K ERNEL OPTIMIZATION FOR DNN I NFERENCE ON MCU S

# # year
# 2024
# the 7 th MLSys Conference
# https://sizezheng.github.io/files/vMCU.pdf

# micro-controller

import numpy as np

def test_definition(param):
  M,N,K, In,W,Out = param
  for m in range(M):
    for n in range(N):
      Out[m,n] = 0
      for k in range(K):
        Out[m,n] += In[m,k]*W[k,n]
  return Out

def test_v1(param):
  M,N,K, In,W,Out = param
  In_L = np.zeros((M,K))
  W_L = np.zeros((K,N))
  Out_L = np.zeros((M,N))

  for m in range(M):
    for n in range(N):
      Out_L[m,n] = 0
      for k in range(K):
        In_L[m,k] = In[m,k]
        W_L[k,n] = W[k,n]
        Out_L[m,n] += In_L[m,k]*W_L[k,n]
      Out[m,n] = Out_L[m,n]
  return Out

def test_v2(param):
  M,N,K, In,W,Out = param
  In_L = np.zeros((M*K))
  W_L = np.zeros((K*N))
  Out_L = np.zeros((M*N))

  for m in range(M):
    for n in range(N):
      Out_L[m*N+n] = 0
      for k in range(K):
        In_L[m*K+k] = In[m,k]
        W_L[k*N+n] = W[k,n]
        Out_L[m*N+n] += In_L[m*K+k]*W_L[k*N+n]
      Out[m,n] = Out_L[m*N+n]
  return Out

def test_v3(param):
  M,N,K, In,W,Out = param
  W_L = np.zeros((K*N))
  # In_L = np.zeros((M*K))
  # Out_L = np.zeros((M*N))
  In_Out_L = np.zeros((M*N + M*K))
  bIn = M*N
  bOut = 0

  for m in range(M):
    for n in range(N):
      In_Out_L[m*N+n + bOut] = 0
      for k in range(K):
        In_Out_L[m*K+k + bIn] = In[m,k]
        W_L[k*N+n] = W[k,n]
        In_Out_L[m*N+n + bOut] += In_Out_L[m*K+k + bIn]*W_L[k*N+n]
      Out[m,n] = In_Out_L[m*N+n + bOut]
  return Out

def test_v4(param):
  M,N,K, In,W,Out = param
  W_L = np.zeros((K*N))
  # In_L = np.zeros((M*K))
  # Out_L = np.zeros((M*N))
  # In_Out_L = np.zeros((M*N + M*K))
  bIn = 1
  bOut = 0
  In_Out_L = np.zeros((bIn + M*K))

  for n in range(N):
    for k in range(K):
      W_L[k*N+n] = W[k,n]
  for m in range(M):
    for k in range(K):
      In_Out_L[m*K+k + bIn] = In[m,k]

  for m in range(M):
    for n in range(N):
      tmp = 0
      for k in range(K):        
        tmp += In_Out_L[m*K+k + bIn]*W_L[k*N+n]
      In_Out_L[m*N+n + bOut] = tmp

  for m in range(M):
    for n in range(N):
      Out[m,n] = In_Out_L[m*N+n + bOut]

  return Out

def test_gemm():
  M,N,K = 2,2,3
  In = np.random.randint(0,10,(M,K))
  W = np.random.randint(0,10,(K,N))
  Out = np.zeros((M,N))

  param = M,N,K, In,W,Out
  Out_g = test_definition(param)
  # Out_t = test_v1(param)
  # Out_t = test_v2(param)
  # Out_t = test_v3(param)
  Out_t = test_v4(param)
  import pdb;pdb.set_trace()
  print(sum(Out_g - Out_t))
  return

if __name__ == "__main__":
  test_gemm()