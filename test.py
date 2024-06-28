import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

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

# Parameters
S_max = 100
K = 40
T = 2
r = 0.22
sigma = 0.55
alpha = 0.35
N = 100
M = 1000
dt = T / M
dS = S_max / N

# Grid initialization
S = np.linspace(0, S_max, N+1)
V = np.maximum(S - K, 0)  # Initial condition

# Crank-Nicolson coefficients
alpha_c = 0.25 * dt * (sigma**2 * S**2 - r * S)
beta = -dt * 0.5 * (sigma**2 * S**2 + r)
gamma = 0.25 * dt * (sigma**2 * S**2 + r * S)

# Tridiagonal matrices
A = np.zeros((N+1, N+1))
B = np.zeros((N+1, N+1))

for i in range(1, N):
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
for j in range(M):
    frac_deriv_V = fractional_derivative(V, dS, alpha, N)
    V_new = np.linalg.solve(A, np.dot(B, V) + dt * frac_deriv_V)
    V_new[0] = 0
    V_new[-1] = S_max - K * np.exp(-r * (T - (j+1) * dt))
    V = V_new

# Interpolating the option value at S=50 (or any other desired value)
S_0 = 50
option_price = np.interp(S_0, S, V)
print(f"Option price at S={S_0}: {option_price}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(S, V, label='Option Price')
plt.axvline(x=S_0, color='r', linestyle='--', label=f'S = {S_0}')
plt.title('Option Price vs Asset Price')
plt.xlabel('Asset Price (S)')
plt.ylabel('Option Price (V)')
plt.legend()
plt.grid(True)
plt.show()
