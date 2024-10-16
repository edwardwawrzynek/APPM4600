#! /usr/bin/env python3
import numpy as np
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
import math

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

    xeval = np.linspace(-1, 1, 1001)
    yeval = eval_monomial(xj, yj, xeval)

    plt.clf()
    plt.plot(xeval, yeval, label="N="+str(N))

    plt.plot(xeval, f(xeval), label="f(x)")
    plt.plot(xj, yj, 'o')
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Monomial Interpolation, N=" + str(N))
    plt.grid()
    plt.legend()
    
    plt.savefig("mono" + str(N) + ".png")

for N in range(2, 20):
    question1b(N)