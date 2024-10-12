import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def question6_2():

    x0 = np.array([0, 0, 0])
    
    Nmax = 100
    tol = 1e-6
    switch_tol = 5e-2
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)

    t = time.time()
    for j in range(50):
      [xstar,g1,ier,its] =  SteepestDescent(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('SD: the error message reads:',ier) 
    print('SD: took this many seconds:',elapsed/50)
    print('SD: number of iterations is:',its)

    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  hybrid(x0,tol,switch_tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('hybrid: the error message reads:',ier) 
    print('hybrid: took this many seconds:',elapsed/50)
    print('hybrid: number of iterations is:',its)
     
     
def evalF(x): 

    F = np.zeros(3)
    
    F[0] = x[0] + np.cos(x[0] * x[1] * x[2]) - 1
    F[1] = (1-x[0])**(0.25) + x[1] + 0.05*(x[2]**2) - 0.15 * x[2] - 1
    F[2] = -x[0]**2 -0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
    return F
    
def evalJ(x): 
    J = np.array([[1-x[1]*x[2]*np.sin(x[0] * x[1] * x[2]),-x[0]*x[2]*np.sin(x[0] * x[1] * x[2]), -x[0]*x[1]*np.sin(x[0] * x[1] * x[2]) ],
    [-0.25*(1-x[0])**(-0.75), 1, 0.1*x[2] - 0.15],
    [-2*x[0], -0.2*x[1] + 0.01, 1]])
    return J

def evalPhi(x):
    f = evalF(x)
    return f[0]**2 + f[1]**2 + f[2]**2

def evalGradPhi(x):
    return np.transpose(evalJ(x)).dot(evalF(x))


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for its in range(Nmax):
        g1 = evalPhi(x)
        z = evalGradPhi(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalPhi(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalPhi(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,ier,its]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalPhi(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalPhi(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,ier,its]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,ier,its]


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]

# Hybrid steepest descent followed by Newton
def hybrid(x0, tol, switch_tol, Nmax):
    [x1, g1, ier1, its1] = SteepestDescent(x0, switch_tol, Nmax)
    if not ier1 == 0:
        return [x1, ier1, its1]
    [x2, ier2, its2] = Newton(x1, tol, Nmax)
    return [x2, ier2, its2]

question6_2()