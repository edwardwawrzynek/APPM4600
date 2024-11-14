#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# APPM 4600 Homework 10
# Edward Wawrzynek

x = np.arange(0, 5, 0.01)
f = np.sin(x)

# 6th order taylor series
t = x - 1.0/6.0*x**3 + 1.0/120.0*x**5
# Pade approximations
p1 = (x - (7.0/60.0)*x**3) / (1 + 1.0/20.0*x**2)
p2 = x / (1 + (1.0/6.0)*x**2 + 7.0/360.0 * x **4)
p3 = (x - (7.0/60.0)*x**3) / (1 + 1.0/20.0*x**2)

plt.plot(x, np.abs(f-t), label="Taylor Approximation")
plt.plot(x, np.abs(f-p1), label="Pade (m=3,n=3)")
plt.plot(x, np.abs(f-p2), label="Pade (m=2,n=4)")
plt.plot(x, np.abs(f-p3), label="Pade (m=4,n=2)")
plt.grid()
plt.xlabel("x")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Approximation Error")
plt.savefig("pade_error.png")

plt.cla()

plt.plot(x, t, label="Taylor Approximation")
plt.plot(x, p1, label="Pade (m=3,n=3)")
plt.plot(x, p2, label="Pade (m=2,n=4)")
plt.plot(x, p3, label="Pade (m=4,n=2)")
plt.plot(x, f, label="f(x)=sin(x)")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Approximation")
plt.savefig("pade.png")