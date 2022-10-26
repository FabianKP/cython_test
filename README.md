Cython Test
===


Testing Cython...

In this little project we test the performance benefit from Cython over
Python by implementing projected gradient descent to solve
non-negative least-squares problems

min_{x >= 0} ||Ax - b||^2.

This is a suitable test case since it has more complexity than e.g.
simple Fibonacci, but it is straightforward to compare.