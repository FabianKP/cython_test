"""
Python implementation of projected gradient descent.
"""

from math import sqrt
import numpy as np


def accelerated_pgd(a: np.array, b: np.array, x0: np.array, num_iter: int):
    """
    Solve the non-negative least-squares problem
        min_x ||A x - b||_2^2 s.t. x >= 0
    using accelerated projected gradient descent.
    Given an initial guess x_0 and 0 < alpha_0 < 1, we set
        y_0 = x_0,
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

    Returns
    -------
    x : shape (n, )
        The minimizer of the NNLS problem.
    rnorm : float
        The residual ||Ax-b||_2.
    """
    # Check input.
    m, n = a.shape
    assert b.shape == (m, )
    assert x0.shape == (n, )
    assert num_iter >= 1
    # Initialize variables
    y = x0
    x_old = x0
    x = x0
    alpha_old = 0.9
    alpha = 0.9
    ata = a.T @ a
    s = np.linalg.norm(ata)
    theta_1 = np.identity(n) - ata / s
    theta_2 = a.T @ b / s
    # Start the iteration.
    for k in range(num_iter):
        print(f"Iteration {k + 1}/{num_iter}", end="")
        x = (theta_1 @ y + theta_2).clip(min=0.)
        alpha = 0.5 * (sqrt(alpha_old ** 4 + 4 * alpha_old ** 2) - alpha_old ** 2)
        beta = alpha_old * (1 - alpha_old) / (alpha_old ** 2 + alpha)
        y = x + beta * (x - x_old)
        x_old = x
        alpha_old = alpha
        print("\r", end="")
    rnorm = np.linalg.norm(a @ x - b)
    return x, rnorm

