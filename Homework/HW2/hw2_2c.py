#! /usr/bin/env python3
import numpy as np

# pertubations
b1 = 2e-5
b2 = 0.7e-5

# b1 = 1e-5
# b2 = 1e-5

# setup system to solve
Ainv = np.array([[1-10**10, 10**10], [1+10**10, -10**10]])
b = np.array([[1], [1]])
bpert = np.array([[b1], [b2]])
# compute exact solution
x = np.matmul(Ainv, b)
# compute perturbed solution
xpert = np.matmul(Ainv, b+bpert)
# absolute error
xdif = x - xpert
# relative error
rel_error = np.linalg.norm(xdif) / np.linalg.norm(x)

print(rel_error)
