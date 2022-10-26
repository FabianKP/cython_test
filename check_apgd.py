
import numpy as np
from time import time

from apgd_cython import accelerated_pgd_cython
m = 4000
n = 3000
num_iter = 1000


# Create random A and b.
A = np.random.randn(m, n)
b = np.random.randn(m)

# Initial guess is zero vector.
x0 = np.zeros(n)

t0 = time()
accelerated_pgd_cython(a=A, b=b, x0=x0, num_iter=num_iter)
t1 = time()

print(f"Accelerated PGD took {t1-t0} seconds.")