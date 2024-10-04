import numpy as np
import math
from numpy.linalg import norm 
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 4*x[1]**2+4*x[2]**2 - 16

def grad_f(x):
    return np.array([2*x[0], 8*x[1], 8*x[1]])

def iterate(f, grad_f, x0, tol, Nmax):
    iters = np.array([x0])
    for n in range(Nmax):
        # update step
        laplacian_x0 = np.sum(grad_f(x0)**2)
        x1 = x0 - f(x0) / laplacian_x0 * grad_f(x0)
        # keep track of iterates
        iters = np.vstack([iters, x1])
        # check for bail out
        if norm(x1 - x0) < tol:
            return [x1, 0, iters]
        
        x0 = x1
    
    return [x1, 1, iters]

def plot_order(x, xstar):
    diff1 = norm(x[1::] - xstar, axis=1)
    diff2 = norm(x[0:-1]-xstar, axis=1)

    plt.plot(np.log(diff2.flatten()), np.log(diff1.flatten()))
    plt.grid()
    plt.xlabel("$\ln || \mathbf{x_{n}} - \mathbf{x^*} ||$");
    plt.ylabel("$\ln || \mathbf{x_{n+1}} - \mathbf{x^*} ||$");

    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    print('the order of the equation is')
    print("lambda = " + str(np.exp(fit[1])))
    print("alpha = " + str(fit[0]))

    alpha = fit[0]
    l = np.exp(fit[1])

    return [fit, alpha, l]


def question3b():
    x0 = np.array([1, 1, 1])
    [xstar, ier, iters] = iterate(f, grad_f, x0, 1e-10, 100)
    print("xstar=", xstar, "ier=", ier)
    plot_order(iters[0:-1], xstar)
    plt.savefig("hw5_3b.png")

question3b()