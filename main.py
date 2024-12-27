import time
from quasi_newton_method import quasi_newton_method_1D
from secant_method import secant_method_1D
import numpy as np
from Rosenbrok_Powell import rosenbrock_grad, powell_quartic_grad, powell_quartic_1d, powell_quartic_1d_grad, rosenbrock_1d, rosenbrock_1d_grad

if __name__ == '__main__':
    # --- Rosenbrock's Function ---
    print("--- Rosenbrock's Function ---")

    # Initial guess for Rosenbrock
    x0_rosen = np.array([-1.2, 1.0])

    # For 1D methods we need to perform Line Search
    # We need to find an descent direction vector d for 1D minimization
    # In this example, I will use d = -grad(x)
    d_rosen = -rosenbrock_grad(x0_rosen)

    # Let's run the methods for Line Search
    initial_alpha = 0.0
    initial_alpha_secant = 0.1

    print("\n-- Quasi-Newton Method --")
    min_alpha, min_val, iterations, cpu_time = quasi_newton_method_1D(lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha),
                                                                      lambda alpha: rosenbrock_1d_grad(
                                                                          x0_rosen, d_rosen, alpha),
                                                                      initial_alpha)

    print(f"Optimal alpha: {min_alpha}, f(x): {
          min_val}, iterations: {iterations}, CPU time: {cpu_time}")

    print("\n-- Secant Method --")
    min_alpha, min_val, iterations, cpu_time = secant_method_1D(lambda alpha: rosenbrock_1d(x0_rosen, d_rosen, alpha),
                                                                initial_alpha, initial_alpha_secant)
    print(f"Optimal alpha: {min_alpha}, f(x): {
          min_val}, iterations: {iterations}, CPU time: {cpu_time}")

    # --- Powell's Quartic Function ---
    print("\n--- Powell's Quartic Function ---")

    # Initial guess for Powell's quartic
    x0_powell = np.array([3.0, -1.0, 0.0, 1.0])

    # For 1D methods we need to perform Line Search
    # We need to find an descent direction vector d for 1D minimization
    # In this example, I will use d = -grad(x)
    d_powell = -powell_quartic_grad(x0_powell)

    # Let's run the methods for Line Search
    initial_alpha = 0.0
    initial_alpha_secant = 0.1

    print("\n-- Quasi-Newton Method --")
    min_alpha, min_val, iterations, cpu_time = quasi_newton_method_1D(lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha),
                                                                      lambda alpha: powell_quartic_1d_grad(
                                                                          x0_powell, d_powell, alpha),
                                                                      initial_alpha)
    print(f"Optimal alpha: {min_alpha}, f(x): {
          min_val}, iterations: {iterations}, CPU time: {cpu_time}")

    print("\n-- Secant Method --")
    min_alpha, min_val, iterations, cpu_time = secant_method_1D(lambda alpha: powell_quartic_1d(x0_powell, d_powell, alpha),
                                                                initial_alpha, initial_alpha_secant)

    print(f"Optimal alpha: {min_alpha}, f(x): {
          min_val}, iterations: {iterations}, CPU time: {cpu_time}")
