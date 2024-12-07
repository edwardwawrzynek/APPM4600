#! /usr/bin/env python3
import numpy as np
import numpy.linalg as la

# APPM 4600 Homework 12
# Power Method
# Edward Wawrzynek

# perform the power method on A
# returns (lambda, eigvector, err, iters)
def power_method(A, relative_tol=1e-15, max_iters=100000):
    # select random b
    b = np.random.random((A.shape[0], 1))
    l = (b.T @ A @ b) / (b.T @ b)
    for n in range(max_iters):
        # apply A
        b1 = A.dot(b)
        # normalize resulting vector
        b1 = b1 / la.norm(b1)
        # calculate the Rayleigh quotient
        l1 = (b1.T @ A @ b1) / (b1.T @ b1)
        # check for convergence
        if (np.abs(l1 - l) / np.abs(l1))[0][0] < relative_tol:
            return (l[0][0], b1, False, n)

        l = l1
        b = b1
    
    return (l[0][0], b, True, max_iters)

# form a hilbert matrix
def hilbert(n):
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            h[i][j] = 1/((i+1) + (j+1) - 1)
    
    return h

def driver():
    print("Question 3(a)")
    for n in range(4, 21, 4):
        A = hilbert(n)
        (l, u, err, iters) = power_method(A, 1e-10)
        print(str(n) + " & " + str(l) + " & "  + str(iters) + " \\\\")

    print("Question 3(b)")
    Ainv = la.inv(hilbert(16))
    (l, u, err, iters) = power_method(Ainv, 1e-16)
    print(1/l)

    print("Question 3(d)")
    A = np.array([[2, 1], [0, 2]])
    (l, u, err, iters) = power_method(A)
    print(err)



driver()