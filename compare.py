import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# Parameters for Fractional Black-Scholes
r = 0.22        # risk-free interest rate
sigma = 0.55    # volatility
beta = 0.35     # fractional order
T = 2           # terminal time
S_max = 100.0   # maximum stock price
N = 100         # number of spatial points
M = 100         # number of time steps
K = 40          # strike price

# Parameters for Fractional Derivative Method
alpha_frac = 0.35   # fractional order for derivative
S_max_frac = 100    # maximum asset price
N_frac = 100        # number of spatial points
M_frac = 1000       # number of time steps

# Function definitions
def fractional_binomial_coeff(alpha, k):
    return sp.gamma(alpha + 1) / (sp.gamma(k + 1) * sp.gamma(alpha - k + 1))

def fractional_derivative(V, h, alpha, N):
    frac_deriv = np.zeros_like(V)
    for i in range(N+1):
        sum = 0
        for k in range(0, i+1):
            if i - k >= 0:
                sum += (-1)**k * fractional_binomial_coeff(alpha, k) * V[i - k]
        frac_deriv[i] = sum / h**alpha
    return frac_deriv

# Initial condition and boundary condition for Fractional Black-Scholes
def initial_condition_black_scholes(S):
    return np.maximum(S - K, 0)

def boundary_condition_black_scholes(S):
    return 0

# Initial condition and boundary condition for Fractional Derivative Method
def initial_condition_frac_deriv(S):
    return np.maximum(S - K, 0)

def boundary_condition_frac_deriv(S):
    return S_max_frac - K * np.exp(-r * T)

# Implicit Euler method for time discretization
def implicit_euler(u_prev, A, delta_t):
    I = np.identity(N+1)
    return np.linalg.solve(I - delta_t * A, u_prev)

# Constructing the stiffness matrix A for FEM (Fractional Black-Scholes)
def construct_stiffness_matrix(S_values, delta_S):
    alpha_fem = 0.5 * sigma**2
    delta_t = T / M
    A = np.zeros((N+1, N+1))
    
    for i in range(1, N):
        A[i, i-1] = -0.5 * delta_t * (r * S_values[i] - alpha_fem * S_values[i]**(2 * beta) / delta_S**2)
        A[i, i] = 1 + delta_t * (r + alpha_fem * S_values[i]**(2 * beta) / delta_S**2)
        A[i, i+1] = -0.5 * delta_t * (-r * S_values[i] - alpha_fem * S_values[i]**(2 * beta) / delta_S**2)
    
    # Boundary conditions
    A[0, 0] = 1
    A[N, N] = 1
    
    return A

# Main solver function for Fractional Black-Scholes
def solve_fractional_black_scholes():
    S_values = np.linspace(0, S_max, N+1)
    delta_S = S_max / N
    A = construct_stiffness_matrix(S_values, delta_S)
    u = initial_condition_black_scholes(S_values)
    delta_t = T / M
    
    for n in range(M):
        u = implicit_euler(u, A, delta_t)
        # Apply boundary condition
        u[0] = boundary_condition_black_scholes(S_values[0])
        u[N] = boundary_condition_black_scholes(S_values[N])
    
    return S_values, u  # Return S_values and option price array

# Main solver function for Fractional Derivative Method
def solve_fractional_derivative():
    # Parameters
    dt = T / M_frac
    dS = S_max_frac / N_frac
    
    # Grid initialization
    S_frac = np.linspace(0, S_max_frac, N_frac+1)
    V_frac = initial_condition_frac_deriv(S_frac)
    
    # Crank-Nicolson coefficients
    alpha_c = 0.25 * dt * (sigma**2 * S_frac**2 - r * S_frac)
    beta = -dt * 0.5 * (sigma**2 * S_frac**2 + r)
    gamma = 0.25 * dt * (sigma**2 * S_frac**2 + r * S_frac)
    
    # Tridiagonal matrices
    A = np.zeros((N_frac+1, N_frac+1))
    B = np.zeros((N_frac+1, N_frac+1))
    
    for i in range(1, N_frac):
        A[i, i-1] = -alpha_c[i]
        A[i, i] = 1 - beta[i]
        A[i, i+1] = -gamma[i]
        B[i, i-1] = alpha_c[i]
        B[i, i] = 1 + beta[i]
        B[i, i+1] = gamma[i]
    
    # Fixing the boundary conditions to ensure the matrix A is non-singular
    A[0, 0] = 1
    A[-1, -1] = 1
    B[0, 0] = 1
    B[-1, -1] = 1
    
    # Time stepping
    for j in range(M_frac):
        frac_deriv_V = fractional_derivative(V_frac, dS, alpha_frac, N_frac)
        V_new = np.linalg.solve(A, np.dot(B, V_frac) + dt * frac_deriv_V)
        V_new[0] = boundary_condition_frac_deriv(S_frac[0])
        V_new[-1] = boundary_condition_frac_deriv(S_frac[-1])
        V_frac = V_new
    
    return S_frac, V_frac  # Return S_frac and option price array

# Solve for option prices
S_values, option_price_black_scholes = solve_fractional_black_scholes()
S_frac, option_price_frac_deriv = solve_fractional_derivative()

# Plotting results
plt.figure(figsize=(14, 8))

# Plot Fractional Black-Scholes Option Price
plt.subplot(2, 1, 1)
plt.plot(S_values, option_price_black_scholes, label='Fractional Black-Scholes Option Price')
plt.title('Fractional Black-Scholes Option Price vs Asset Price')
plt.xlabel('Asset Price (S)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)

# Plot Fractional Derivative Method Option Price
plt.subplot(2, 1, 2)
plt.plot(S_frac, option_price_frac_deriv, label='Fractional Derivative Method Option Price')
plt.title('Fractional Derivative Method Option Price vs Asset Price')
plt.xlabel('Asset Price (S)')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting Payoff Diagrams
plt.figure(figsize=(10, 6))

# Plot Payoff Diagram (max(S-K, 0))
payoff = np.maximum(S_values - K, 0)
plt.plot(S_values, payoff, label='Payoff at Expiry (K=40)', linestyle='--')

plt.title('Payoff Diagram Comparison')
plt.xlabel('Asset Price (S)')
plt.ylabel('Payoff')
plt.legend()
plt.grid(True)
plt.show()
