import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = np.arange(1, 0.01)

print("the first three values of x are ", x[0], x[1], x[2])

w = 10 ** (-np.linspace(1,10,10))
x = np.linspace(1,10,10)

s = 3*w

plt.semilogy(x, w, label="W")
plt.semilogy(x,s, label="S")
plt.xlabel("X")
plt.ylabel("W")
plt.legend()
plt.show()
