import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

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

def compute_order_ndim(x, xstar):
    diff1 = norm(x[1::] - xstar, axis=1)
    diff2 = norm(x[0:-1]-xstar, axis=1)
    fit = np.polyfit(np.log(diff2.flatten()), np.log(diff1.flatten()),1)
    print('the order of the equation is')
    print("lambda = " + str(np.exp(fit[1])))
    print("alpha = " + str(fit[0]))

    alpha = fit[0]
    l = np.exp(fit[1])

    return [fit, alpha, l]


def evalF(xn): 
    return np.array([3*xn[0]**2 - xn[1]**2, 3*xn[0]*xn[1]**2 - xn[0]**3 - 1])
    
def evalJ(x): 
    return np.array([[6*x[0], -2*x[1]], 
        [3*x[1]**2 - 3**x[0]**2, 6*x[0]*x[1]]])


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    iters = np.array([x0])
    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       iters = np.vstack([iters, x1])

       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its, iters]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its,iters]

def question1_1(Nmax):
    xn = np.array([1, 1])
    x = np.zeros((Nmax, 2))
    for n in range(Nmax):
        x[n] = xn
        xn = xn - np.array([[1/6, 1/18], [0, 1/6]]).dot(np.array(
            [3*xn[0]**2 - xn[1]**2, 
            3*xn[0]*xn[1]**2 - xn[0]**3 - 1]))
    
    print("Question 1(a)")
    print(xn[0])
    print(xn[1])
    print(x)
    compute_order_ndim(x, xn)

def question1_3():
    [xstar, ier, its, iters] = Newton(np.array([1,1]), 1e-10, 100)
    print("Question 1(c)")
    print("xstar=", xstar, "ier=", ier, "its=", its)
    print(xstar[0])
    print(xstar[1])
    print(iters)
    compute_order_ndim(iters[0:-1], xstar)

question1_1(30)
question1_3()