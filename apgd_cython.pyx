"""
Cython implementation of projected gradient descent.
"""

from libc.math cimport sqrt
import numpy as np
cimport numpy as np

ctypedef np.float64_t FLOAT


cpdef np.ndarray[FLOAT, ndim=1] accelerated_pgd_cython(np.ndarray[FLOAT, ndim=2] a,
                                                       np.ndarray[FLOAT, ndim=1] b,
                                                       np.ndarray[FLOAT, ndim=1] x0,
                                                       int num_iter):
    """
    Solve the non-negative least-squares problem
        min_x ||A x - b||_2^2 s.t. x >= 0
    using accelerated projected gradient descent.
    Given an initial guess x_0 and 0 < alpha_0 < 1, we set
        y_0 = x_0,        y_0 = x_0,

        s = ||A.T A||,
        theta_1 = Id - A.T A / s,
        theta_2 = A.T b / s.
    Then the algorithm iterates
        x_{k+1} = [theta_1 y_k + theta_2]^+,
        alpha_{k+1} = 0.5 * (sqrt(alpha_k^4 + 4  alpha_k^2) - alpha_k^2),
        beta_{k+1} = alpha_k * (1 - alpha_k) / (alpha_k^2 + alpha_{k+1}),
        y_{k+1} = x_{k+1} + beta_k * (x_{k+1} - x_{k}).

    Parameters
    ----------
    a : shape (m, n)
        The matrix A.
    b : shape (m, )
        The right-hand side b.
    x0 : shape (n, )
        Initial guess for the minimizer.
    num_iter : int
        Number of iterations.

    Returns    assert b.shape == (m, )
    assert x0.shape == (n, )
    assert num_iter >= 1
    -------
    x : shape (n, )
        The minimizer of the NNLS problem.
    """
    # Check input.
    cdef Py_ssize_t m = a.shape[0]
    cdef Py_ssize_t n = a.shape[1]
    # Initialize variables
    cdef np.ndarray[FLOAT, ndim=1] y = x0
    cdef np.ndarray[FLOAT, ndim=1] x_old = x0
    cdef np.ndarray[FLOAT, ndim=1] x
    x = x0
    cdef float alpha_old = 0.9
    cdef float alpha = 0.9
    cdef np.ndarray[FLOAT, ndim=2] ata = a.T @ a
    cdef float s = np.linalg.norm(ata)
    cdef np.ndarray[FLOAT, ndim=2] theta_1
    theta_1 = np.identity(n, dtype=np.float64) - ata / s
    cdef np.ndarray[FLOAT, ndim=1] theta_2 = a.T @ b / s
    # Start the iteration.
    for k in range(num_iter):
        x = (theta_1 @ y + theta_2).clip(min=0.)
        alpha = 0.5 * (sqrt(alpha_old ** 4 + 4 * alpha_old ** 2) - alpha_old ** 2)
        beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 + alpha)
        y = x + beta * (x - x_old)
        x_old = x
        alpha_old = alpha
    return x