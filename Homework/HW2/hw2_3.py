#! /usr/bin/env python3
import numpy as np
import math

x = 9.999999995000000e-10

# naive computation
y = math.e ** x - 1
print("naive y: ", y)

y_taylor = x + x**2/2 
print("taylor y: ", f'{y_taylor:.27}')