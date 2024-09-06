#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['text.usetex'] = True

def run(x):
    delta = np.arange(-16, 1, 1)
    delta = 10.0**delta

    # two different evaluations for the same function
    #eval2 = np.cos(x + delta) - np.cos(x)
    eval1 = -delta * np.sin(x) - np.square(delta)/2 * np.cos(x)
    eval2 = -2*np.sin((2*x+delta)/2)*np.sin(delta/2)

    # absolute difference
    diff = np.abs(eval1 - eval2)
    
    plt.loglog(delta, diff, label="x="+str(x))
    plt.xlabel("$\delta$")
    plt.ylabel("Absolute Error")
    plt.title("Error between evaluation of (1) and (2) [our algorithm]")
    plt.legend()

run(math.pi)
run(10**6)
plt.savefig("hw1_5_our_algorithm.png")
plt.show()