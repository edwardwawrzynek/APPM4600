#! /usr/bin/env python3
import numpy as np

# APPM 4600, Homework 11, Problem 1a
# Edward Wawrzynek

def eval_composite_trap(M,a,b,f):
  h = (b-a)/M
  x = np.arange(0, M+1)*h + a
  # trapezoidal weights
  w = h*np.ones(M+1)
  w[0] = h*0.5
  w[-1] = h*0.5

  I = np.sum(f(x) * w)

  return I, x, w

def eval_composite_simpsons(M,a,b,f):
  # force M even
  M = int(M/2)*2

  h = (b-a)/(M)
  x = np.arange(0, (M)+1)*h + a
  w = np.ones((M)+1)
  for i in range(M):
    w[2*i+1] = 4
    if i < M-1:
      w[2*i+2] = 2

  w = w * h/3

  I = np.sum(f(x) * w)
  return I, x, w

def driver():
  f = lambda x: 1/(1 + x**2)

  method = eval_composite_trap
  a = -5
  b = 5
  N = 1000
  I, _, _ = method(N, a, b, f)

  print(I)
  
driver()