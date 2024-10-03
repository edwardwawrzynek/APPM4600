#! /usr/bin/env python3

import numpy as np
import scipy.special
import math
import matplotlib.pyplot as plt

# function of interest
def f(x):
    return np.exp(3*x) - 27*(x**6) + 27*(x**4)*np.exp(x) - 9*(x**2)*np.exp(2*x)

# derivative of f
def df(x):
    return 3*(np.exp(x) - 6*x)*(np.exp(x) - 3*x**2)**2


def g(x):
  return f(x) / df(x)

def dg(x):
  return (6*x**2 + np.exp(x)*(x**2-4*x+2)) / ((np.exp(x) - 6*x)**2)


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
        p = np.append(p, p0)
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]

def compute_order(x, xstar):
    diff1 = np.abs(x[1::] - xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    print('the order of the equation is')
    print("lambda = " + str(np.exp(fit[1])))
    print("alpha = " + str(fit[0]))

    alpha = fit[0]
    l = np.exp(fit[1])

    return [fit, alpha, l]

tol = 1e-10

# find root via Newton
[iters, xstar1, ier, it] = newton(f, df, 1, 4, tol, 100)
print("xstar1=", xstar1, ", ier=", ier, "iters=", iters)
compute_order(iters, xstar1)

# find root via modified Newton (ii)
[iters, xstar1, ier, it] = newton(g, dg, 1, 4, tol, 100)
print("xstar1=", xstar1, ", ier=", ier, "iters=", iters)
compute_order(iters, xstar1)

# find root via modified Newton (iii)
[iters, xstar1, ier, it] = newton(f, df, 3, 3, tol, 100)
print("xstar1=", xstar1, ", ier=", ier, "iters=", iters)
compute_order(iters, xstar1)