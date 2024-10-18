import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv

def driver():
  f = lambda x: np.exp(x)
  a = 0
  b = 1
  # create points you want to evaluate at#
  Neval = 100
  xeval = np.linspace(a,b,Neval)
  # number of intervals#
  Nint = 10
  #evaluate the linear spline#
  yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
  # evaluate f at the evaluation points#
  fex = f(xeval)
  plt.figure()
  plt.plot(xeval,fex,'ro-')
  plt.plot(xeval,yeval,'bs-')
  plt.legend()
  plt.show()
  err = abs(yeval-fex)
  plt.figure()
  plt.plot(xeval,err,'ro-')
  plt.show()

def question32(N):
    i = np.linspace(1, N, N)
    xj = -1 + (i - 1) * 2 / (N-1)
    # evaluate f(xj)
    f = lambda xj: 1/(1 + (10*xj)**2)
    yj = f(xj)

    xeval = np.linspace(-1, 1, 1001)
    yeval = eval_lin_spline(xeval, 1001, -1, 1, f, N)

    plt.clf()
    plt.plot(xeval, yeval, label="N="+str(N))

    plt.plot(xeval, f(xeval), label="f(x)")
    plt.xlabel("x")
    plt.ylabel("p(x)")
    plt.title("Linear Spline Interpolation, N=" + str(N))
    plt.grid()
    plt.legend()
    
    plt.savefig("lin" + str(N) + ".png")

    plt.clf()
    plt.semilogy(xeval, np.abs(f(xeval) - yeval))
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title("Linear Spline Error, N=" + str(N))
    plt.grid()
    plt.legend()
    plt.savefig("lin_error" + str(N) + ".png")

def eval_line(x, a1, fa1, b1, fb1):
  return (x-a1) * (fb1-fa1) / (b1 - a1) + fa1

def eval_lin_spline(xeval,Neval,a,b,f,Nint):
  #create the intervals for piecewise approximations#
  xint = np.linspace(a,b,Nint+1)
  #create vector to store the evaluation of the linear splines#
  yeval = np.zeros(Neval)
  for jint in range(Nint):
    #find indices of xeval in interval (xint(jint),xint(jint+1))#
    #let ind denote the indices in the intervals#
    atmp = xint[jint]
    btmp= xint[jint+1]

    # find indices of values of xeval in the interval
    ind= np.where((xeval >= atmp) & (xeval <= btmp))
    xloc = xeval[ind]
    n = len(xloc)
    #temporarily store your info for creating a line in the interval of interest#
    fa = f(atmp)
    fb = f(btmp)
    yloc = np.zeros(len(xloc))

    for kk in range(n):
      #use your line evaluator to evaluate the spline at each location
      yloc[kk] = eval_line(xloc[kk], atmp, fa, btmp, fb)#Call your line evaluator with points (atmp,fa) and (btmp,fb)

    # Copy yloc into the final vector
    yeval[ind] = yloc
  
  return yeval

question32(4)
question32(10)
question32(20)