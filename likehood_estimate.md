# probability
- conditional parameter -> result
- P(x | theta)

# likehood
- result -> conditional parameter
- L(theta | x)

# maximum likehood estimate
- result -> the best conditional parameter
- issue: what the theta is equal to, can obtain the most likely result

# example_1
- coin: 0, 1
- 10 sample
  - 1: 7 times
  - 0: 3 times
- L(theta) = (theta)^7 * (1-theta)^3

# example_2
- coin: 1, 2
- 5 sample: 1 1 2 1 2
  - 1: 3 times
  - 0: 2 times
- L(theta) = (theta)^3 * (1-theta)^2
- problem solving process
  - logarithmic: Ln(lhs) = Ln(rhs)
    - Ln(L(theta)) = Ln((theta)^3 * (1-theta)^2)
    - Ln(L(theta)) = 3 * Ln(theta) + 2 * Ln(1-theta)
  - derivation
    - d(Ln(L(theta)))/d(theta) = (3/theta) - (2/(1-theta)) = 0
    - theta = 3/5

# example_3
- X~U(0,a), a  is unknown
- f(x) = 1/a, (0,a)
- f(x) = 0, others
- sample: x1, x2, ..., xn
  - in the continuous system distribution, the probability of each point is 0
  - and the random variable at each point has a probability density 1/a
    - f(x1), f(x2), ..., f(xn) = 1/a, 1/a, ..., 1/a
  - the joint probability density of this group of samples L(a) = (1/a)^n
- problem solving process
  - to make L(a) maximum, that is, to make a a^n minimum
  - because x1,x2,...,xn are all extracted, so they all meet the range(0,a), so the maximum likehood a cannot be smaller than them
  - the maximum likehood a = max(x1,x2,...,xn)


