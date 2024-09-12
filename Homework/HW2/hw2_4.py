#! /usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# part (a)
t = np.arange(0, math.pi+1e-10, math.pi/30)
y = np.cos(t)

# compute the sum of elementwise multiplication of terms
S = np.sum(np.multiply(t, y))
print("the sum is:", S)

# part (b)
def plot_b_parametric(theta, R, dr, f, p):
    x = R*(1 + dr*np.sin(f*theta))*np.cos(theta)
    y = R*(1 + dr*np.sin(f*theta))*np.sin(theta)
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("y")

theta = np.arange(0, 2*math.pi, 1e-2)
plot_b_parametric(theta, 1.2, 0.1, 15, 0)
plt.savefig("hw2_3_b1.png")
plt.close()

for i in range(10):
    R = i
    dr = 0.05
    f = 2 + i
    p = random.uniform(0, 2)
    plot_b_parametric(theta, R, dr, f, p)

plt.savefig("hw2_3_b2.png")

