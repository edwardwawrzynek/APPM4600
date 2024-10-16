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

# construct the vandermonde matrix for the given data points
def construct_vandermonde(xj):
    N = xj.size-1
    V = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            V[j][i] = xj[j]**i
    
    return V

# perform monomial interpolation of (xj, yj) on xeval
def eval_monomial(xj, yj, xeval):
    V = construct_vandermonde(xj)
    coeff = inv(V) @ yj

    yeval = coeff[0]*np.ones(xeval.shape)

    for j in range(1, xj.size):
        for i in range(xeval.size):
            yeval[i] = yeval[i] + coeff[j]* (xeval[i]**j)

    return yeval

def question1b(N):
    i = np.linspace(1, N, N)
    xj = -1 + (i - 1) * 2 / (N-1)
    # evaluate f(xj)
    f = lambda xj: 1/(1 + (10*xj)**2)
    yj = f(xj)

    xeval = np.linspace(-0.25, 0.25, 1001)
    yeval_mono = eval_monomial(xj, yj, xeval)
    yeval_bary = eval_bary_lagrange(xj, yj, xeval)

    plt.clf()
    plt.plot(xeval, yeval_mono, label="Monomial")
    plt.plot(xeval, yeval_bary, label="Barycentric Lagrange")

    #plt.plot(xeval, f(xeval), label="f(x)")
    plt.plot(xj, yj, 'o')
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Barycentric Lagrange vs Monomial Interpolation, N=" + str(N))
    plt.grid()
    plt.legend()
    
    plt.show()

question1b(50)