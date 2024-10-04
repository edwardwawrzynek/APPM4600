#! /usr/bin/env python3

import numpy as np
import scipy.special
import math
import matplotlib.pyplot as plt

# function of interest
def f(x):
    return x**6 - x - 1

# derivative of f
def df(x):
    return 6*x**5-1

#perform newton's method (provided)
def newton(f,fp,m,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
    f,fp - function and derivative
    p0 - initial guess for root
    tol - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
    Returns:
    p - an array of the iterates
    pstar - the last iterate
    info - success message
    - 0 if we met tol
    - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-m*f(p0)/fp(p0)
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
        p = np.append(p, p0)
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def secant(f, p0, p1, tol, Nmax):
  p = np.zeros(2)
  p[0] = p0
  p[1] = p1
  for i in range(Nmax):
    # secant step
    p2 = p1 - f(p1) * (p1 - p0) / (f(p1) - f(p0))

    # check tolerance
    if (abs(p2-p1) < tol):
      return [p, p2, 0, it]
    # update values
    p = np.append(p, p2)
    p0 = p1
    p1 = p2
  
  return [p, p2, 1, it]


def plot_order(x, xstar):
    diff1 = np.abs(x[1::] - xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    plt.plot(np.log(diff2.flatten()), np.log(diff1.flatten()))

tol = 1e-16

# find root via Newton
print("Newton")
[iters1, xstar1, ier, it] = newton(f, df, 1, 2, tol, 100)
print("xstar1=", xstar1, ", ier=", ier, "iters=", iters1)
for i in range(len(iters1)):
  print(i+1, " & ", np.abs(iters1[i] - xstar1), "\\\\")

# find root via secant
print("Secant")
[iters2, xstar2, ier, it] = secant(f, 2, 1, tol, 100)
print("xstar1=", xstar2, ", ier=", ier, "iters=", iters2)
for i in range(len(iters2)):
  print(i+1, " & ", np.abs(iters2[i] - xstar2), "\\\\")

plot_order(iters1, xstar1)
plot_order(iters2, xstar2)
plt.legend(["Newton", "Secant"])
plt.xlabel("$log(x_k - \\alpha)$")
plt.ylabel("$log(x_{k+1} - \\alpha)$")
plt.savefig("hw4_5.png")