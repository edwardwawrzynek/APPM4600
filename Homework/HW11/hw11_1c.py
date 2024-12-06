#! /usr/bin/env python3
import numpy as np
import scipy
import scipy.integrate

# APPM 4600, Homework 11, Problem 1c
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
  for i in range(int(M/2)):
    w[2*i+1] = 4
    if i < int(M/2)-1:
      w[2*i+2] = 2

  w = w * h/3

  I = np.sum(f(x) * w)
  return I, x, w

def driver():
  f = lambda x: 1/(1 + x**2)

  a = -5
  b = 5
  I_exact, _ = scipy.integrate.quad(f, a, b, epsabs=1e-12)
  I_trap, _, _ = eval_composite_trap(1290, a, b, f)
  I_simp, _, _ = eval_composite_simpsons(108, a, b, f)

  print("Trapezoidal error:")
  print(np.abs(I_exact - I_trap))

  print("Simpsons' error:")
  print(np.abs(I_exact - I_simp))

  I1, _, info1 = scipy.integrate.quad(f, a, b, epsabs=1e-4, full_output=True)
  print("quadpack iters 10^-4:")
  print(info1['neval'])

  I2, _, info2 = scipy.integrate.quad(f, a, b, epsabs=1e-6, full_output=True)
  print("quadpack iters 10^-4:")
  print(info2['neval'])
  
driver()