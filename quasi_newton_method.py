import time


def quasi_newton_method_1D(f, f_prime, x0, tol=1e-5, max_iter=1000):
    """Finds the minimizer using Quasi-Newton method (1D)."""
    x = x0
    iterations = 0
    start_time = time.time()
    h = 1e-5
    # Approximate Hessian using finite difference
    hessian = (f_prime(x + h) - f_prime(x))/h

    while abs(f_prime(x)) > tol and iterations < max_iter:
        x = x - f_prime(x) / hessian  # Update x based on approximated Hessian
        iterations += 1
    end_time = time.time()
    cpu_time = end_time - start_time
    return x, f(x), iterations, cpu_time
