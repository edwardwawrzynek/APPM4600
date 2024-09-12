# import libraries
import numpy as np

def question3():
     funcs = [
          lambda x: x * (1 + (7-x**5)/(x**2))**3,
          lambda x: x - (x**5-7)/(x**2),
          lambda x: x - (x**5-7)/(5*x**4),
          lambda x: x - (x**5-7)/12
     ]
     tol = 1e-10
     # verify that x=7^1/5 is fixed point
     for f in funcs:
          x = 7**(1/5)
          [xstar, ier] = fixedpt(f, x, tol, 100)
          if np.abs(x-xstar) < tol:
               print("x=7^1/5 is fixed point, x=", x, ", xstar=", xstar, ", ier=", ier)
          else:
               print("x=7^1/5 is not fixed point, x=", x, ", xstar=", xstar, ", ier=", ier)
     
     # apply fixed point iteration with x0=1
     for f in funcs:
          x = 1
          try:
               [xstar, ier] = fixedpt(f, x, tol, 100)
               print("x=", x, ", xstar=", xstar, ", ier=", ier)
          except:
               print("overflow error, ")

# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

question3()