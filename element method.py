import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.22        # risk-free interest rate
sigma = 0.55     # volatility
beta = 0.35      # fractional order
T = 2     # terminal time
S_max = 100.0   # maximum stock price
N = 100         # number of spatial points
M = 100         # number of time steps
K = 40          # strike price

# Spatial discretization
S_values = np.linspace(0, S_max, N+1)
delta_S = S_max / N

# Time discretization
delta_t = T / M
t_values = np.linspace(0, T, M+1)

# Initial condition
def initial_condition(S):
    return np.maximum(S - K, 0)

# Boundary conditions
def boundary_condition(S):
    return 0

# Implicit Euler method for time discretization
def implicit_euler(u_prev, A, delta_t):
    I = np.identity(N+1)
    return np.linalg.solve(I - delta_t * A, u_prev)

# Constructing the stiffness matrix A for FEM
def construct_stiffness_matrix():
    alpha = 0.5 * sigma**2
    A = np.zeros((N+1, N+1))
    
    for i in range(1, N):
        A[i, i-1] = -0.5 * delta_t * (r * S_values[i] - alpha * S_values[i]**(2 * beta) / delta_S**2)
        A[i, i] = 1 + delta_t * (r + alpha * S_values[i]**(2 * beta) / delta_S**2)
        A[i, i+1] = -0.5 * delta_t * (-r * S_values[i] - alpha * S_values[i]**(2 * beta) / delta_S**2)
    
    # Boundary conditions
    A[0, 0] = 1
    A[N, N] = 1
    
    return A

# Main solver function
def solve_fractional_black_scholes():
    A = construct_stiffness_matrix()
    u = initial_condition(S_values)
    
    for n in range(M):
        u = implicit_euler(u, A, delta_t)
        # Apply boundary condition
        u[0] = boundary_condition(S_values[0])
        u[N] = boundary_condition(S_values[N])
    
    return u

# Solve for option price
option_price = solve_fractional_black_scholes()

# Plotting results
plt.figure(figsize=(10, 6))

# Plot option price
plt.plot(S_values, option_price, label='Option Price')

# Plot payoff diagram (max(S-K, 0))
payoff = np.maximum(S_values - K, 0)
plt.plot(S_values, payoff, label='Payoff at Expiry (K=50)', linestyle='--')

plt.title('Fractional Black-Scholes Option Price and Payoff')
plt.xlabel('Stock Price (S)')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()
