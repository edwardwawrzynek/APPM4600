#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy.linalg as scila
import time

data = np.array([
[100, 0.012948989868164062, 0.0009374618530273438 ],
[500, 0.22867417335510254, 0.0067484378814697266 ],
[1000, 1.8918616771697998, 0.021845102310180664 ],
[2000, 6.13384485244751, 0.13550925254821777 ],
[4000, 38.78315711021423, 0.6089046001434326 ],
[5000, 56.77767086029053, 0.9785189628601074 ],
])

N = data[:,0]
solve = data[:,1]
lu = data[:,2]

plt.loglog(N, solve, label="scipy.linalg.solve")
plt.loglog(N,lu, label="LU solve")
plt.xlabel("N")
plt.ylabel("Time to run 50 solves [s]")
plt.title("Time to solve 50 square systems")
plt.grid()
plt.legend()

plt.show()



print(N)