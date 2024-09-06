#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**9 -18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 -4032*x**4 +5376*x**3 - 4608*x** 2 + 2304*x - 512

def f_p(x):
    return (x - 2) ** 9

x = np.arange(1.920, 2.080, 0.001)
# plot f via expansion

#plt.plot(x, f(x))
plt.plot(x, f_p(x))
plt.xlabel("x")
plt.ylabel("p(x)")
plt.savefig("hw1_1_eval2.png")