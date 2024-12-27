import numpy as np

# --- Benchmark Functions from Document ---
# Rosenbrock's function and its gradient


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

# Powell's quartic function


def powell_quartic(x):
    return (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[3])**4

# For Powell's quartic, finding the gradient analytically is complex; let's use finite differences


def powell_quartic_grad(x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (powell_quartic(x_plus_h) - powell_quartic(x)) / h
    return grad


# Function for 1D minimization with Line Search using Rosenbrock Function
def rosenbrock_1d(x, d, alpha):
    x_new = x + alpha*d
    return rosenbrock(x_new)


def rosenbrock_1d_grad(x, d, alpha):
    h = 1e-5
    return (rosenbrock_1d(x, d, alpha+h)-rosenbrock_1d(x, d, alpha))/h

# Function for 1D minimization with Line Search using Powell's Function


def powell_quartic_1d(x, d, alpha):
    x_new = x + alpha*d
    return powell_quartic(x_new)


def powell_quartic_1d_grad(x, d, alpha):
    h = 1e-5
    return (powell_quartic_1d(x, d, alpha+h)-powell_quartic_1d(x, d, alpha))/h
