#! /usr/bin/env python3

import numpy as np
import scipy.special
import math
import matplotlib.pyplot as plt

Ti = 20
Ts = -15
a = 0.138e-6
t = 60 * 24 * 60 * 60

# function of interest
def f(x):
    return (Ti - Ts) * scipy.special.erf(x / (2*np.sqrt(a*t))) + Ts

# derivative of f
def df(x):
    return (Ti-Ts) * np.exp(-np.square(x) / (4*a*t)) / (np.sqrt(math.pi * a * t))

# perform bisection (provided)
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when relative interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a) > tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]

#perform newton's method (provided)
def newton(f,fp,p0,tol,Nmax):
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
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

# plot f
t_sample = np.arange(0, 1, 0.1)
plt.plot(t_sample, f(t_sample))
plt.xlabel("x (depth) [m]")
plt.ylabel("f(x) (temperature after 60 days) [C]")
plt.savefig("hw4_1.png")

# find root via bisection
tol = 1e-13
[xstar, ier] = bisection(f, 0, 1, tol)
print("xstar=", xstar, ", ier=", ier)

# find root via Newton
[iters, xstar1, ier, it] = newton(f, df, 0.01, tol, 100)
print("xstar1=", xstar1, ", ier=", ier)

# newton starting at x=10
[iters, xstar2, ier, it] = newton(f, df, 10, tol, 100)
print("xstar2=", xstar2, ", ier=", ier)