import time
import numpy as np
import math
import sympy as sp

# --- Helper Functions ---


def approximate_gradient(func, x, h=1e-5):
    """Approximates the gradient of a function using finite differences."""
    grad = np.zeros_like(x, dtype=float)  # Ensure float type
    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (func(x_plus_h) - func(x)) / h
    return grad

# --- Fibonacci Method ---


def fibonacci_method_1d(f, interval, epsilon):
    Fs = [1, 1]
    max_fibonacci_number = (interval[1] - interval[0]) / epsilon
    while Fs[-1] < max_fibonacci_number:
        Fs.append(Fs[-1] + Fs[-2])
    Fs.pop()
    k = len(Fs)

    a, b = interval
    x1 = a + (Fs[k - 3] / Fs[k - 1]) * (b - a)
    x2 = a + (Fs[k - 2] / Fs[k - 1]) * (b - a)

    iterations = 0
    start_time = time.time()
    for i in range(k - 2):
        iterations += 1
        f_x1 = f(x1)
        f_x2 = f(x2)
        if f_x1 > f_x2:
            a = x1
            x1 = x2
            x2 = a + (Fs[k - 2] / Fs[k - 1]) * (b - a)
        else:
            b = x2
            x2 = x1
            x1 = a + (Fs[k - 3] / Fs[k - 1]) * (b - a)
        k -= 1
        if b - a <= epsilon:
            break
    end_time = time.time()

    minimum = (a + b) / 2
    f_min = f(minimum)
    cpu_time = end_time - start_time
    return minimum, f_min, iterations, cpu_time

# --- Golden Section Method ---


def golden_section_method_1d(f, interval, epsilon):
    golden_ratio = (math.sqrt(5) - 1) / 2
    a, b = interval
    x1 = a + (1 - golden_ratio) * (b - a)
    x2 = a + golden_ratio * (b - a)

    f_x1 = f(x1)
    f_x2 = f(x2)

    iterations = 0
    start_time = time.time()
    while (b - a) > epsilon:
        iterations += 1
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            x1 = a + (1 - golden_ratio) * (b - a)
        elif f_x1 > f_x2:
            a = x1
            x1 = x2
            x2 = a + golden_ratio * (b - a)
        f_x1 = f(x1)
        f_x2 = f(x2)

    end_time = time.time()
    minimum = (a + b) / 2
    f_min = f(minimum)
    cpu_time = end_time - start_time
    return minimum, f_min, iterations, cpu_time

# --- Newton's Method ---


def newtons_method_1d(f, f_prime, f_double_prime, x0, tol=1e-5, max_iter=1000):
    x = x0
    old_x = 0
    iterations = 0
    start_time = time.time()

    while abs(x - old_x) > tol and iterations < max_iter:
        iterations += 1
        first_deriv = f_prime(x)
        second_deriv = f_double_prime(x)
        old_x = x
        x = x - (first_deriv/second_deriv)
    end_time = time.time()
    f_min = f(x)
    cpu_time = end_time - start_time
    return x, f_min, iterations, cpu_time

# --- Quasi-Newton Method (1D) ---


def quasi_newton_method_1d(f, f_prime, x0, tol=1e-5, max_iter=1000, h=1e-5):
    x = x0
    iterations = 0
    start_time = time.time()

    # Approximate Hessian using finite difference
    hessian = (f_prime(x + h) - f_prime(x))/h

    while abs(f_prime(x)) > tol and iterations < max_iter:
        x = x - f_prime(x) / hessian  # Update x based on approximated Hessian
        iterations += 1
    end_time = time.time()
    cpu_time = end_time - start_time
    return x, f(x), iterations, cpu_time

# --- Secant Method (1D) ---


def secant_method_1d(f, x0, x1, tol=1e-5, max_iter=1000):
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

# --- Benchmark Functions ---


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])


def powell_quartic(x):
    return (x[0] + 10 * x[1])**2 + 5 * (x[2] - x[3])**2 + (x[1] - 2 * x[2])**4 + 10 * (x[0] - x[3])**4


def powell_quartic_grad(x):
    return approximate_gradient(powell_quartic, x)


def rosenbrock_1d(x, d, alpha):
    x_new = x + alpha*d
    return rosenbrock(x_new)


def rosenbrock_1d_grad(x, d, alpha):
    h = 1e-5
    return (rosenbrock_1d(x, d, alpha+h)-rosenbrock_1d(x, d, alpha))/h


def powell_quartic_1d(x, d, alpha):
    x_new = x + alpha*d
    return powell_quartic(x_new)


def powell_quartic_1d_grad(x, d, alpha):
    h = 1e-5
    return (powell_quartic_1d(x, d, alpha+h)-powell_quartic_1d(x, d, alpha))/h


# --- Main Execution ---
if __name__ == "__main__":
    # --- Initialization ---
    a = 0
    b = 3
    interval = [a, b]
    EPSILON = 0.01
    bench_marks = {}

   # Helper function for our 1D minimisation problem
    def f_1d_problem(x):
        return 2 * (x)**2 - 5 * x + 3

    # --- 1D Minimization Tests---
    print("--- 1D Minimization on f(x) = 2x^2 - 5x + 3 ---")
    # -- Fibonacci Method --
    minimum, f_min, iterations, cpu_time = fibonacci_method_1d(
        f_1d_problem, interval, EPSILON)
    print(f"Fibonacci Method - Min: {minimum:.3f}, f(min): {
          f_min:.3f}, Iterations: {iterations}, Time: {cpu_time:.6f}")
    bench_marks["Fibonacci"] = {"Iterations": iterations, "TimeTaken": round(
        cpu_time, 6), "F": round(f_min, 3), "X": round(minimum, 3)}

    # -- Golden Section Method --
    minimum, f_min, iterations, cpu_time = golden_section_method_1d(
        f_1d_problem, interval, EPSILON)
    print(f"Golden Section Method - Min: {minimum:.3f}, f(min): {
          f_min:.3f}, Iterations: {iterations}, Time: {cpu_time:.6f}")
    bench_marks["Golden"] = {"Iterations": iterations, "TimeTaken": round(
        cpu_time, 6), "F": round(f_min, 3), "X": round(minimum, 3)}

    # -- Newton's Method --
    xSymbol = sp.symbols('x')
    newtonF_x = 2 * (xSymbol)**2 - 5 * xSymbol + 3
    dF_x_1 = sp.lambdify(xSymbol, sp.diff(newtonF_x, xSymbol))
    dF_x_2 = sp.lambdify(xSymbol, sp.diff(
        sp.diff(newtonF_x, xSymbol), xSymbol))

    minimum, f_min, iterations, cpu_time = newtons_method_1d(
        f_1d_problem, dF_x_1, dF_x_2, 2, tol=EPSILON)
    print(f"Newton's Method - Min: {minimum:.3f}, f(min): {
          f_min:.3f}, Iterations: {iterations}, Time: {cpu_time:.6f}")
    bench_marks["Newton"] = {"Iterations": iterations, "TimeTaken": round(
        cpu_time, 6), "F": round(f_min, 3), "X": round(minimum, 3)}

    # --- 2D Minimization with Line Search ---

    # --- Rosenbrock Function ---
    print("\n--- 2D Minimization on Rosenbrock with Line Search ---")
    x0_rosen = np.array([-1.2, 1.0], dtype=float)
    d_rosen = -rosenbrock_grad(x0_rosen)
    initial_alpha = 0.0
    initial_alpha_secant = 0.1
    interval_rosen = [0, 3]

    # Fibonacci
    min_alpha, min_val, iterations, cpu_time = fibonacci_method_1d(
        lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha), interval_rosen, EPSILON)
    print(f"Fibonacci (Rosenbrock) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Golden Section
    min_alpha, min_val, iterations, cpu_time = golden_section_method_1d(
        lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha), interval_rosen, EPSILON)
    print(f"Golden Section (Rosenbrock) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Quasi-Newton
    min_alpha, min_val, iterations, cpu_time = quasi_newton_method_1d(
        lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha),
        lambda alpha: rosenbrock_1d_grad(x0_rosen, d_rosen, alpha),
        initial_alpha
    )
    print(f"Quasi-Newton (Rosenbrock) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

   # Secant
    min_alpha, min_val, iterations, cpu_time = secant_method_1d(
        lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha),
        initial_alpha, initial_alpha_secant
    )
    print(f"Secant (Rosenbrock)- Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Newton
    xSymbol = sp.symbols('alpha')
    rosenbrock_x = rosenbrock_1d(x0_rosen, d_rosen, xSymbol)
    dF_x_1 = sp.lambdify(xSymbol, sp.diff(rosenbrock_x, xSymbol))
    dF_x_2 = sp.lambdify(xSymbol, sp.diff(
        sp.diff(rosenbrock_x, xSymbol), xSymbol))

    min_alpha, min_val, iterations, cpu_time = newtons_method_1d(
        lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha), dF_x_1, dF_x_2, 2, tol=EPSILON)
    print(f"Newton (Rosenbrock) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, Time: {cpu_time:.6f}")

    # --- Powell's Quartic Function ---
    print("\n--- 2D Minimization on Powell's Quartic with Line Search ---")
    x0_powell = np.array([3.0, -1.0, 0.0, 1.0], dtype=float)
    d_powell = -powell_quartic_grad(x0_powell)
    initial_alpha = 0.0
    initial_alpha_secant = 0.1
    interval_powell = [0, 3]

    # Fibonacci
    min_alpha, min_val, iterations, cpu_time = fibonacci_method_1d(
        lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha), interval_powell, EPSILON)
    print(f"Fibonacci (Powell) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Golden Section
    min_alpha, min_val, iterations, cpu_time = golden_section_method_1d(
        lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha), interval_powell, EPSILON)
    print(f"Golden Section (Powell) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Quasi-Newton
    min_alpha, min_val, iterations, cpu_time = quasi_newton_method_1d(
        lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha),
        lambda alpha: powell_quartic_1d_grad(x0_powell, d_powell, alpha),
        initial_alpha
    )
    print(f"Quasi-Newton (Powell) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Secant
    min_alpha, min_val, iterations, cpu_time = secant_method_1d(
        lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha),
        initial_alpha, initial_alpha_secant
    )
    print(f"Secant (Powell)- Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, CPU time: {cpu_time:.6f}")

    # Newton
    xSymbol = sp.symbols('alpha')
    powell_x = powell_quartic_1d(x0_powell, d_powell, xSymbol)
    dF_x_1 = sp.lambdify(xSymbol, sp.diff(powell_x, xSymbol))
    dF_x_2 = sp.lambdify(xSymbol, sp.diff(sp.diff(powell_x, xSymbol), xSymbol))

    min_alpha, min_val, iterations, cpu_time = newtons_method_1d(
        lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha), dF_x_1, dF_x_2, 2, tol=EPSILON)
    print(f"Newton (Powell) - Optimal alpha: {min_alpha:.6f}, f(x): {
          min_val:.6f}, Iterations: {iterations}, Time: {cpu_time:.6f}")

    # --- Benchmarks for all ---
    print("\n# --- Benchmarks for all Methods ---")
    print(bench_marks)
