import numpy as np
from mem import *

# N,K = 16,16
N,K = 1,11
TK = 4

def test_Listing_1_serial_summation(A, B):
  # A serial summation of 11 elements over a one-dimensional domain k
  # f() += in(r)
  for n in range(N):
    B[n] = 0
    for k in range(K):
      B[n] = B[n] + A[n,k]
  return

def test_Listing_1_split(A, B):
  # The split scheduling directive reshapes the domain into a two-dimensional reduction. The reduction is still serial
  # f.pslit(r,rx,ry,4)
  K_Iteration = (K+TK-1)//TK
  print(K_Iteration)
  for n in range(N):
    B[n] = 0
    for ko in range(K_Iteration):
      for ki in range(TK):
        k = ko*TK + ki
        if k < K:
          B[n] = B[n] + A[n,k]
  return

def test_Listing_1_parallel_ko(A, B):
  # The rfactor directive creates and returns a new (orange square) intermediate stage which computes partial results along each row. 
  # The intermediate stage is now data parallel over its outer loop, which is indexed with the new pure variable v. 
  # The merge stage (blue hexagon) retains only the serial loop over the reduction variable ry. We have reassociated the summation.
  # f.rfactor(ry,v).parallel(v)
  mem = MEM()
  K_Iteration = (K+TK-1)//TK
  print(K_Iteration)
  B_rf = mem.new([K_Iteration, N], "zero", dtype=np.int32)
  for n in range(N):
    B[n] = 0
    for ko in range(K_Iteration):
      B_rf[ko, n] = 0
      for ki in range(TK):
        k = ko*TK + ki
        if k < K:
          B_rf[ko, n] = B_rf[ko, n] + A[n,k]
    for ko in range(K_Iteration):
      B[n] += B_rf[ko, n]
  return

def test_Listing_1_parallel_ki(A, B):
  # Alternatively, one can use rfactor to make the inner loop data parallel. This can be used to vectorize reductions. 
  # The intermediate stage computes sums of whole vectors, data parallel across the vector lanes, and then the merge stage sums the lanes of the vector result. 
  # The strategies in 3a and 3b can be combined to both vectorize and parallelize a summation. 
  # Note that 3b requires the reduction to be commutative as it changes the order of computation.
  # f.rfactor(ry,v).parallel(v)
  mem = MEM()
  K_Iteration = (K+TK-1)//TK
  print(K_Iteration)
  B_rf = mem.new([TK, N], "zero", dtype=np.int32)
  for n in range(N):
    B[n] = 0
    for ki in range(TK):
      B_rf[ki, n] = 0
      for ko in range(K_Iteration):
        k = ko*TK + ki
        if k < K:
          B_rf[ki, n] = B_rf[ki, n] + A[n,k]
    for ki in range(TK):
      B[n] += B_rf[ki, n]
  return

def test_Listing_1():
  mem = MEM()
  A = mem.new([N,K], "rand", dtype=np.int32)
  B = mem.new([N], "zero", dtype=np.int32)

  # test_Listing_1_serial_summation(A, B)
  # test_Listing_1_split(A, B)
  # test_Listing_1_parallel_ko(A, B)
  test_Listing_1_parallel_ki(A, B)

  import pdb;pdb.set_trace()
  return

def test_Listing_3():
  # two-dimensional convolution, parallelizable across Vars x and y
  mem = MEM()
  Y,X,RY,RX = 3,16,3,3
  blur = mem.new([X,Y], "zero", dtype=np.int32)
  k = mem.new([RX,RY], "rand", dtype=np.int32)
  in_ = mem.new([X+RX,Y+RY], "rand", dtype=np.int32)

  # first stage
  for y in range(Y):
    for x in range(X):
      blur[x, y] = 0
  # second update stage
  for y in range(Y):
    for x in range(X):
      for ry in range(RY):
        for rx in range(RX):
          blur[x, y] += k[rx, ry] * in_[x-rx, y-ry]
  return

def test_Listing_4_compute_root(params, hist, input):
  X,RX,RY = params
  # first stage
  for x in range(X):
    hist[x] = 0
  # second update stage
  for ry in range(RY):
    for rx in range(RX):
      hist[input[rx,ry]] += 1
  return

def test_Listing_4_compute_parallel_y(params, hist, input):
  X,RX,RY = params
  mem = MEM()
  # first stage
  for x in range(X):
    hist[x] = 0
  # intermediate stage
  # computes the partial histogram over each row of the input
  intm = mem.new([X,RY], "zero", dtype=np.int32)

  # for y in range(RY): # parallel, bind
  #   for rx in range(RX):
  #     intm[input[rx,y], y] += 1
  #   # merge stage
  #   for x in range(X):
  #     hist[x] += intm[x,y]

  # vectorize(x,4)
  for y in range(RY): # parallel, bind
    for rx in range(RX):
      intm[input[rx,y], y] += 1
  # merge stage
  # for y in range(RY): 
    for xo in range(X//4):
      hist[xo*4:(xo+1)*4] += intm[xo*4:(xo+1)*4,y]

  return

def test_Listing_4():
  # Computing the histogram of an image is harder to parallelize since its update stage only involves serial RVars
  mem = MEM()
  X = 256 # image gray-level:0~255
  RX,RY = 720,576
  hist = mem.new([X], "zero", dtype=np.int32)
  input = mem.new([RX,RY], "rand", dtype=np.int8, upper=256)
  params = X,RX,RY
  # test_Listing_4_compute_root(params, hist, input)
  test_Listing_4_compute_parallel_y(params, hist, input)

  import pdb;pdb.set_trace()
  return

def input(rx, ry):
  return rx*rx + ry*ry

def test_circle_1(params):
  radius, restrict, max_val, rx_,ry_ = params
  for rx in range(radius):
    for ry in range(radius):
      if (rx*rx + ry*ry <=restrict):
        if input(rx, ry) > max_val:
          rx_,ry_ = rx,ry
          max_val = max(max_val, input(rx, ry))
  print(rx_, ry_, max_val)
  return

def test_circle_2(params):
  radius, restrict, max_val, rx_,ry_ = params
  mem = MEM()
  intm = mem.new([radius], "zero", dtype=np.int32)

  # for y in range(radius):
  #   for rx in range(radius):
  #     if (rx*rx + y*y <=restrict):
  #       intm[y] = max(intm[y], input(rx, y))
  # import pdb;pdb.set_trace()
  # max_val = 0
  # for ry in range(radius):
  #   max_val = max(max_val, intm[ry])

  max_val = 0
  for y in range(radius): # parallel, bind
    for rx in range(radius):
      if (rx*rx + y*y <=restrict):
        intm[y] = max(intm[y], input(rx, y))
    max_val = max(max_val, intm[y])

  print(rx_, ry_, max_val)
  return

def test_circle():
  radius = 10
  restrict = 100
  max_val, rx_,ry_ = 0,0,0
  params = radius, restrict, max_val, rx_,ry_
  # test_circle_1(params)
  test_circle_2(params)

  return

if __name__ == "__main__":
  # test_Listing_1()
  # test_Listing_3()
  # test_Listing_4()
  test_circle()
