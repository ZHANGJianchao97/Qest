#!/usr/bin/env python
# coding: utf-8


#by ZHANGJianchao from UEC

import numpy as np
from scipy.linalg import sqrtm

def update_kraus(K, M, rho, drho, FI_inv, prob, M_deriv, alpha=0.01, W=None):
    """
    Update Kraus operators K using Fisher information-based rule.

    Parameters:
    - K: list of Kraus operators (numpy arrays of shape [d, d])
    - M: list of POVM elements (M[k] = K[k]† K[k])
    - rho: density matrix
    - drho: list of derivatives of rho w.r.t. parameters
    - FI_inv: inverse Fisher information matrix (numpy array shape [num_params, num_params])
    - prob: list of probabilities Tr[M_k rho] for each k
    - M_deriv: numpy array of shape [num_params, num_povm], M_deriv[i, k] = Tr[M_k drho[i]]
    - alpha: update step size
    - W: optional weight matrix, default identity

    Returns:
    - new_K: updated list of Kraus operators
    - new_M: updated list of POVM elements
    """
    d = rho.shape[0]
    num_povm = len(M)
    num_params = len(drho)

    if W is None:
        W = np.eye(num_params)

    # Construct Rho^i = sum_j FI_inv[j, i] * drho[j]
    Rho = [sum(FI_inv[j, i] * drho[j] for j in range(num_params)) for i in range(num_params)]

    # D matrix is M_deriv @ FI_inv
    D = FI_inv @ M_deriv  # shape [num_params, num_povm]

    # l[:, k] = D[:, k] / prob[k]
    l = np.array([
        D[:, k] / prob[k] if prob[k] > 1e-8 else np.zeros(num_params)
        for k in range(num_povm)
    ]).T  # shape [num_params, num_povm]

    # Construct Y_k
    Y = []
    for k in range(num_povm):
        Yk = np.zeros((d, d), dtype=np.complex128)
        for i in range(num_params):
            for j in range(num_params):
                term = 2 * Rho[i] * l[j, k] - rho * l[i, k] * l[j, k]
                Yk += term * W[i, j]
        Y.append(Yk)

    # Compute Lambda
    Lambda = np.zeros((d, d), dtype=np.complex128)
    for k in range(num_povm):
        Lambda += Y[k].conj().T @ K[k].conj().T @ K[k] + K[k].conj().T @ K[k] @ Y[k]
    Lambda *= 0.5

    # Update Kraus operators
    new_K = np.zeros((num_povm, d, d), dtype=np.complex128)
    new_M = np.zeros((num_povm, d, d), dtype=np.complex128)
    for k in range(num_povm):
        H = K[k] @ (Y[k] - Lambda)
        newK = K[k] + alpha * H
        new_K[k]=newK
        new_M[k]=newK.conj().T @ newK

    # Normalize
    G = sum(new_M)
    if np.linalg.det(G) < 1e-10:
        print("Warning: Determinant of G is very small.")
    G_inv_sqrt = sqrtm(np.linalg.inv(G))

    for k in range(num_povm):
        new_K[k] = new_K[k] @ G_inv_sqrt
        new_M[k] = new_K[k].conj().T @ new_K[k]

    return new_K, new_M


# In[2]:


import numpy as np
from scipy.linalg import sqrtm

def initialize_kraus_operators(d, num_povm):
    K = np.random.randn(num_povm, d, d) + 1j * np.random.randn(num_povm, d, d)
    return K

def build_povm_from_kraus(K):
    num_povm = K.shape[0]
    M = np.zeros_like(K, dtype=np.complex128)
    for i in range(num_povm):
        M[i] = K[i].conj().T @ K[i]

    # Normalize to ensure completeness: ∑M_i = I
    M_sum = np.sum(M, axis=0)
    Msuminv_sqrt = sqrtm(np.linalg.inv(M_sum))
    for i in range(num_povm):
        K[i] = K[i] @ Msuminv_sqrt  # Update K to normalized form
        M[i] = K[i].conj().T @ K[i]
    return M, K

# 计算费舍尔信息矩阵（使用 drho）
def fisher_information(M, rho, drho, reg_lambda=1e-6):
    num_povm = M.shape[0]
    dim = len(drho) # dim: num of paras
    F = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(num_povm):
        M_i = M[i]
        P_i = np.trace(M_i @ rho)
        if np.abs(P_i) < 1e-8:
            continue
        dlogP = np.array([
            np.trace(M_i @ drho[j]) / P_i for j in range(dim)
        ], dtype=np.complex128)
        F += np.outer(dlogP, dlogP.conj()) * P_i
    #F += reg_lambda * np.eye(dim)
    return F.real

# 目标函数：Tr(F^-1)
def objective_function(M, rho, drho, reg_lambda=1e-6):
    F = fisher_information(M, rho, drho, reg_lambda=reg_lambda)
    if np.linalg.cond(F) < 1e6:
        return np.trace(np.linalg.inv(F))
    else:
        return np.inf

def compute_prob_and_derivatives(M, rho, drho):
    """
    Compute the measurement probabilities and their derivatives.

    Parameters:
        M (list or np.ndarray): List of POVM elements M_k (shape: num_povm x d x d).
        rho (np.ndarray): Density matrix (shape: d x d).
        drho (list of np.ndarray): List of derivatives ∂ρ/∂θ_i (length: num_params, each shape: d x d).

    Returns:
        prob (np.ndarray): Vector of probabilities Tr(M_k rho), shape: (num_povm,)
        D (np.ndarray): Matrix of derivatives Tr(M_k ∂ρ/∂θ_i), shape: (num_params, num_povm)
    """
    num_povm = len(M)
    num_params = len(drho)
    d_rho = rho.shape[0]

    prob = np.zeros(num_povm)
    D = np.zeros((num_params, num_povm))

    for k in range(num_povm):
        pk = np.trace(M[k] @ rho)
        if np.imag(pk) > 1e-5:
            print(f"Warning: Imag part of prob[{k}] is large: {np.imag(pk)}. Truncating.")
        prob[k] = np.real(pk)

        for i in range(num_params):
            val = np.trace(M[k] @ drho[i])
            if np.imag(val) > 1e-5:
                print(f"Warning: Imag part of D[{i},{k}] is large: {np.imag(val)}. Truncating.")
            D[i, k] = np.real(val)

    return prob, D
    


def optimize_kraus(K, rho, drho, steps=100, reg_lambda=1e-6):
    M, K = build_povm_from_kraus(K)
    loss_list = []
    alpha_list = [10**i for i in range(-4, 4)]  # 1e-4 to 1e1

    for step in range(steps):
        F = fisher_information(M, rho, drho, reg_lambda=reg_lambda)
        F_inv = np.linalg.inv(F)
        prob, M_deriv = compute_prob_and_derivatives(M, rho, drho)

        # Line search for best alpha
        K, best_alpha, loss = line_search_step(K, M, rho, drho, F_inv, prob, M_deriv, alpha_list, reg_lambda)
        M, _ = build_povm_from_kraus(K)
        loss_list.append(loss)

        print(f"Step {step+1}, Loss: {loss:.6f}, Best α: {best_alpha:.1e}")

    return M, loss_list


#initial rho drho in M copies

def tensor_n(mat, n):
    result = mat
    for _ in range(n - 1):
        result = np.kron(result, mat)
    return result

def drho_tensor(rho, drho_list, M):
    rhoM = tensor_n(rho, M)
    dM = len(drho_list)
    dim = rho.shape[0]
    
    drhoM = [np.zeros((dim**M, dim**M), dtype=complex) for _ in range(dM)]

    for k in range(dM):
        for i in range(M):
            parts = []
            for j in range(M):
                if j == i:
                    parts.append(drho_list[k])
                else:
                    parts.append(rho)
            term = parts[0]
            for p in parts[1:]:
                term = np.kron(term, p)
            drhoM[k] += term

    return rhoM, drhoM

def line_search_step(K, M, rho, drho, F_inv, prob, M_deriv, alpha_list, reg_lambda):
    best_alpha = None
    best_loss = np.inf
    best_K = None

    for alpha in alpha_list:
        K_new, M_new = update_kraus(K.copy(), M.copy(), rho, drho, F_inv, prob, M_deriv, alpha=alpha, W=None)
        M_temp, _ = build_povm_from_kraus(K_new)
        loss = objective_function(M_temp, rho, drho, reg_lambda)

        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_K = K_new

    return best_K, best_alpha, best_loss



# In[3]:


def initialize_kraus_operators(d, num_povm):
    K = np.random.randn(num_povm, d, d) + 1j * np.random.randn(num_povm, d, d)
    return K

def build_povm_from_kraus(K):
    num_povm = K.shape[0]
    M = np.zeros_like(K, dtype=np.complex128)
    for i in range(num_povm):
        M[i] = K[i].conj().T @ K[i]

    # Normalize to ensure completeness: ∑M_i = I
    M_sum = np.sum(M, axis=0)
    Msuminv_sqrt = sqrtm(np.linalg.inv(M_sum))
    for i in range(num_povm):
        K[i] = K[i] @ Msuminv_sqrt  # Update K to normalized form
        M[i] = K[i].conj().T @ K[i]
    return M, K

# 计算费舍尔信息矩阵（使用 drho）
def fisher_information(M, rho, drho, reg_lambda=1e-6):
    num_povm = M.shape[0]
    dim = len(drho) # dim: num of paras
    F = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(num_povm):
        M_i = M[i]
        P_i = np.trace(M_i @ rho)
        if np.abs(P_i) < 1e-8:
            continue
        dlogP = np.array([
            np.trace(M_i @ drho[j]) / P_i for j in range(dim)
        ], dtype=np.complex128)
        F += np.outer(dlogP, dlogP.conj()) * P_i
    #F += reg_lambda * np.eye(dim)
    return F.real

# 目标函数：Tr(F^-1)
def objective_function(M, rho, drho, reg_lambda=1e-6):
    F = fisher_information(M, rho, drho, reg_lambda=reg_lambda)
    if np.linalg.cond(F) < 1e6:
        return np.trace(np.linalg.inv(F))
    else:
        return np.inf

def optimize_kraus_grad_search(K, rho, drho, steps=100, reg_lambda=1e-6):
    alpha_list = [10**i for i in range(-4, 2)]
    d = K.shape[1]
    loss_list = []

    for step in range(steps):
        M, K = build_povm_from_kraus(K)
        grad = np.zeros_like(K, dtype=np.complex128)

        for i in range(K.shape[0]):
            for a in range(d):
                for b in range(d):
                    K_perturbed = K.copy()
                    K_perturbed[i, a, b] += 1e-6
                    M1, _ = build_povm_from_kraus(K_perturbed)
                    loss1 = objective_function(M1, rho, drho, reg_lambda)

                    K_perturbed[i, a, b] -= 2e-6
                    M2, _ = build_povm_from_kraus(K_perturbed)
                    loss2 = objective_function(M2, rho, drho, reg_lambda)

                    grad[i, a, b] = (loss1 - loss2) / (2e-6)

        best_loss = np.inf
        best_alpha = None
        best_K = None

        for alpha in alpha_list:
            K_try = K - alpha * grad
            M_try, _ = build_povm_from_kraus(K_try)
            loss_try = objective_function(M_try, rho, drho, reg_lambda)

            if loss_try < best_loss:
                best_loss = loss_try
                best_alpha = alpha
                best_K = K_try

        K = best_K
        loss_list.append(best_loss)
        print(f"[GradDescent] Step {step+1}, Loss: {best_loss:.6f}, Best α: {best_alpha:.1e}")

    return build_povm_from_kraus(K)[0], loss_list


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import time

# ===== SETUP =====
num_trials = 1  # You can change this to 10 or more
max_plot_steps = 100  # Only show first N steps on plot
d = 2
num_copy = 2
num_povm = 4

# Pauli matrices
sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma3 = np.array([[1, 0], [0, -1]])

rho = (np.eye(2)) / 2
drho_list = [0.5 * sigma1, 0.5 * sigma2]

# Expand to M copies
rhoM, drhoM = drho_tensor(rho, drho_list, num_copy)

all_losses_my = []
all_losses_grad = []

# ===== RUN MULTIPLE TRIALS =====
for trial in range(num_trials):
    print(f"\nTrial {trial+1}/{num_trials}")
    
    K_init = initialize_kraus_operators(pow(d, num_copy), num_povm)

    # My algorithm
    start = time.time()
    _, losses_my = optimize_kraus(K_init.copy(), rhoM, drhoM, steps=max_plot_steps)
    all_losses_my.append(losses_my)
    time_my = time.time() - start

    # Gradient method
    start = time.time()
    _, losses_grad = optimize_kraus_grad_search(K_init.copy(), rhoM, drhoM, steps=max_plot_steps)
    all_losses_grad.append(losses_grad)
    time_grad = time.time() - start

# ===== PLOTTING =====
x_range = range(max_plot_steps)
plt.figure(figsize=(10, 6))

# Plot My Algorithm curves
for i in range(num_trials):
    plt.plot(x_range, all_losses_my[i], label=f"My Algo Run {i+1}" if i == 0 else "", 
             color='tab:blue', marker='o', markersize=3, linewidth=1.5)

# Plot Gradient method curves
for i in range(num_trials):
    plt.plot(x_range, all_losses_grad[i], label=f"Grad Desc Run {i+1}" if i == 0 else "", 
             color='tab:orange', marker='s', markersize=3, linewidth=1.5)

plt.xlabel("Step")
plt.ylabel("Loss (Tr[F⁻¹])")
plt.yscale("log")
plt.title(f"Convergence Comparison over {num_trials} Trials (First {max_plot_steps} Steps)")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save the figure
#plt.savefig(f"comparison_{num_trials}_runs.png", dpi=300)
plt.show()





