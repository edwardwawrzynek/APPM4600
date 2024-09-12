# import libraries
import numpy as np
import math

def question1():
  f = lambda x: x*x*(x-1)
  a = -1
  b = 2

  tol = 1e-12
  [astar, ier] = bisection(f, a, b, tol)
  print("Root: ", astar, ", Error: ", ier)

def print_bisection(f, a, b, tol):
  [astar, ier] = bisection(f, a, b, tol)
  print("Root: ", astar, ", Error: ", ier)

def question2():
  f1 = lambda x: (x-1)*(x-3)*(x-5)
  f2 = lambda x: (x-1)*(x-1)*(x-3)
  f3 = lambda x: np.sin(x)

  tol = 1e-5
  print_bisection(f1, 0, 2.4, tol)
  print_bisection(f2, 0, 2, tol)
  print_bisection(f3, 0, 0.1, tol)
  print_bisection(f3, 0.5, 3*math.pi/4, tol)


# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

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
    while (abs(d-a)> tol):
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

question1()    
question2()      

