"""
Quick demo of solving random NNLS problem with accelerated PGD.
"""


import numpy as np
# Import Lawson-Hanson method which we will use as a benchmark.
from scipy.optimize import nnls
from time import time

# Import Vanilla-Python implementation.
from src.apgd import accelerated_pgd
# Import Cython-implementation.
from apgd_cython import accelerated_pgd_cython


m = 4000
n = 3000
num_iter = 1000


# Create random A and b.
A = np.random.randn(m, n)
b = np.random.randn(m)

# Initial gues is one vector.
x0 = np.ones(n)

# Solve min_{x >= 0} ||Ax - b||^2 with accelerated PGD.
t0 = time()
print("Solving with accelerated PGD:")
x_pgd, rnorm_pgd = accelerated_pgd(a=A, b=b, x0=x0, num_iter=num_iter)
t_pgd = time() - t0

# Solve with Cython-version
t0 = time()
print("Solving with Cython implementation:")
x_cython = accelerated_pgd_cython(a=A, b=b, x0=x0, num_iter=num_iter)
t_cython = time() - t0

# For benchmark, solve with Lawson-Hanson method.
t0 = time()
print("Solving with Lawson-Hanson method:")
x_lh, rnorm_lh = nnls(A, b)
t_lh = time() - t0

err_pgd = np.linalg.norm(x_pgd - x_lh) / np.linalg.norm(x_lh)

# Print evaluation:
print(f"Lawson-Hanson took {t_lh} s. Residual norm: {rnorm_lh}.")
print(f"Accelerated PGD took {t_pgd} s. Residual norm: {rnorm_pgd}. Relative error: {err_pgd}.")
print(f"Cython implementation took {t_cython}.")

