#! /usr/bin/env python3
import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math

# evaluate w_j for xj
def eval_wj(xj):
    wj = np.zeros(xj.size)
    for j in range(xj.size):
        w = 1
        for i in range(xj.size):
            if i != j:
                w *= (xj[j] - xj[i])
        
        wj[j] = 1/w
    
    return wj

def bary_lagrange(xj, yj, wj, x):
    if x in xj:
        i = np.where(xj == x)
        return yj[i[0]]

    n = xj.size
    num = np.sum( wj / (x*np.ones(n) - xj) * yj)
    denom = np.sum(wj / (x*np.ones(n) - xj))

    return num / denom

def eval_bary_lagrange(xj, yj, xeval):
    wj = eval_wj(xj)

    yeval = np.zeros(xeval.size)
    for n in range(xeval.size):
        yeval[n] = bary_lagrange(xj, yj, wj, xeval[n])

    return yeval

def question1b(N):
    i = np.linspace(1, N, N)

    xj = np.cos((2*i-1)*math.pi / (2*N))

    # evaluate f(xj)
    f = lambda xj: 1/(1 + (10*xj)**2)
    yj = f(xj)

    xeval = np.linspace(-1, 1, 1001)
    yeval = eval_bary_lagrange(xj, yj, xeval)

    plt.clf()
    plt.plot(xeval, yeval, label="N="+str(N))

    plt.plot(xeval, f(xeval), label="f(x)")
    plt.plot(xj, yj, 'o')
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Chebyshev Points, N=" + str(N))
    plt.grid()
    plt.legend()
    
    plt.savefig("cheb" + str(N) + ".png")

for N in range(2, 93, 2):
    question1b(N)