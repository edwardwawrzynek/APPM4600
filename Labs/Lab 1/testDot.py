import numpy as np
import numpy.linalg as la
import math
import time

def driver():
    n = 100
    x = np.linspace(0,np.pi,n)

    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    f = lambda x: np.sin(x)
    g = lambda x: np.cos(x)
    y = f(x)
    w = g(x)
    
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    
    # print the output
    print('the dot product is : ', dp)

    # matrix vector multiplication test
    m = np.array([[1,2],[3,4]])
    v = np.array([[1],[2]])
    c = matrixVectorMultiply(m,v)
    print("The matrix vector product is ", c)

    # multiply large matrices
    m_big = np.random.random_sample((100,100))
    v_big = np.random.random_sample((100,1))
    c_big = matrixVectorMultiply(m_big, v_big)
    print("The matrix vector product is ", c_big)

    # time 1000 of our multiplication
    st = time.time()
    for i in range(1000):
        m_big = np.random.random_sample((100,100))
        v_big = np.random.random_sample((100,1))
        c_big = matrixVectorMultiply(m_big, v_big)
    end = time.time()
    print("our matrix multiply: ", end - st)

    # time 1000 of numpy's multiplication
    st = time.time()
    for i in range(1000):
        m_big = np.random.random_sample((100,100))
        v_big = np.random.random_sample((100,1))
        c_big = np.matmul(m_big, v_big)
    end = time.time()
    print("np's matrix multiply: ", end - st)



    return

def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

# multiply matrix a by column vector v
def matrixVectorMultiply(a,v):
    assert(a.shape[1] == v.shape[0])
    assert(v.shape[1] == 1)

    result = np.ndarray(shape=(a.shape[0],1))
    for i in range(a.shape[0]):
        elem = 0.
        for j in range(a.shape[1]):
            elem = elem + a[i][j] * v[j][0]
        result[i][0] = elem
    
    return result

driver()
