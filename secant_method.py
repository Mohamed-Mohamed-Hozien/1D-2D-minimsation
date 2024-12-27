import time


def secant_method_1D(f, x0, x1, tol=1e-5, max_iter=1000):
    """Finds the minimizer using the Secant method (1D)."""
    start_time = time.time()
    x_prev = x0
    x_curr = x1
    iterations = 0
    while abs(f(x_curr) - f(x_prev)) > tol and iterations < max_iter:

        x_next = x_curr - f(x_curr) * (x_curr - x_prev) / \
            (f(x_curr) - f(x_prev))  # Secant formula
        x_prev = x_curr
        x_curr = x_next
        iterations += 1
    end_time = time.time()
    cpu_time = end_time - start_time
    return x_curr, f(x_curr), iterations, cpu_time
