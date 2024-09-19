#! /usr/bin/env python3
# APPM 4600 Homework 3 Question 5

import numpy as np
import matplotlib.pyplot as plt
import math

def question5_a():
  x = np.arange(-10, 10, 0.01)
  y = x - 4*np.sin(2*x)-3

  plt.plot(x, y)
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.grid()
  plt.savefig("hw3_5_a.png")

def question5_b():
  rtol = 0.5e-10
  
  f = lambda x: -np.sin(2*x) + 5*x/4 - 3/4

  # attempt fixed point iteration starting at the approximate location of all the roots
  for guess in [-0.898, -0.55, 1.732, 3.16, 4.517]:
    [xstar, x_guess, ier] = fixedpt(f, guess, rtol, 100)
    print("xstar=", xstar)
    print("ier=", ier)
    print("number of iterations=", len(x_guess))

# run fixed point
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x_guess = np.zeros(0)

    count = 0
    while (count <Nmax):
        x_guess = np.append(x_guess, x0)
        count = count +1
        x1 = f(x0)
        if (abs(x1-x0)/abs(x1) <tol):
            xstar = x1
            ier = 0
            return [xstar,x_guess,ier]
        x0 = x1

    xstar = x1
    ier = 1
    return [xstar, x_guess, ier]

question5_a()
question5_b()