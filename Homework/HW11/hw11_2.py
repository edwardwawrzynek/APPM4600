#! /usr/bin/env python3
import numpy as np
import scipy
import scipy.integrate

# APPM 4600, Homework 11, Problem 2
# Edward Wawrzynek

def eval_composite_simpsons(M,a,b,f):
  # force M even
  M = int(M/2)*2

  h = (b-a)/(M)
  x = np.arange(0, (M)+1)*h + a
  w = np.ones((M)+1)
  for i in range(int(M/2)):
    w[2*i+1] = 4
    if i < int(M/2)-1:
      w[2*i+2] = 2

  print(w)
  w = w * h/3

  I = np.sum(f(x) * w)
  return I, x, w

def driver():
  f = lambda t: t*np.cos(1/t)

  a = 1e-12
  b = 1
  N = 4
  I, _, _ = eval_composite_simpsons(N, a, b, f)

  I_exact, _ = scipy.integrate.quad(f, a, b)
  

  print((I - I_exact) / I_exact)

  print(I)
  print(I_exact)
  
driver()