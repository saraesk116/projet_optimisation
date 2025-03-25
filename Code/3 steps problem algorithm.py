import numpy as np
from scipy.optimize import linprog

# Paramètres du problème
T = 3
c_vectors = [np.array([1, 2]), np.array([3, 1]), np.array([1, 1])]
A_matrices = [np.array([[2, -1], [1, 3]]), np.array([[3, 2], [1, 4]]), np.array([[1, 2], [2, 1]])]
b_vectors = [np.array([1, 2]), np.array([4, 5]), np.array([3, 6])]
E_matrices = [np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 3]])]

# Résolution globale de référence
A_global = np.block([
    [A_matrices[0], np.zeros((2, 4))],
    [E_matrices[0], A_matrices[1], np.zeros((2, 2))],
    [np.zeros((2, 2)), E_matrices[1], A_matrices[2]]
])
result_global = linprog(np.concatenate(c_vectors), A_ub=-A_global, b_ub=-np.concatenate(b_vectors), method='highs')
print(f"Solution globale optimale: {result_global.x}\n")

def solve_stage(t, prev_x=None, cuts=[]):
    """Résout une étape avec coupes et contraintes de liaison."""
    c = c_vectors[t]
    A = A_matrices[t]
    b = b_vectors[t].copy()
    
    # Ajustement pour les contraintes de liaison
    if t > 0 and prev_x is not None:
        b -= E_matrices[t-1] @ prev_x
    
    # Ajout des coupes
    if cuts:
        A_cut = np.array([cut[0] for cut in cuts])
        b_cut = np.array([cut[1] for cut in cuts])
        A = np.vstack([A, A_cut])
        b = np.hstack([b, b_cut])
    
    res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')
    if not res.success:
        raise ValueError(f"Échec étape {t}: {res.message}")
    return res.x, res.ineqlin.marginals

# Algorithme de Benders
cutting_planes = [[] for _ in range(T)]
z_sup = np.inf
z_inf = -np.inf
epsilon = 0.1  # Relaxation pour l'exemple
history = []

for it in range(10):
    # Forward pass
    x_hat = []
    for t in range(T):
        x_prev = x_hat[t-1] if t > 0 else None
        x_t, _ = solve_stage(t, x_prev, cutting_planes[t])
        x_hat.append(x_t)
    
    z_sup_new = sum(c @ x for c, x in zip(c_vectors, x_hat))
    history.append(("Forward", z_sup_new))
    
    # Backward pass - Génération des coupes
    new_cuts = [[] for _ in range(T)]
    for t in range(T-1, 0, -1):
        _, pi = solve_stage(t, x_hat[t-1])  # Résolution duale
        ai = pi @ E_matrices[t-1]
        bi = pi @ b_vectors[t]
        new_cuts[t-1].append((ai, bi))
    
    # Mise à jour des coupes
    for t in range(T):
        cutting_planes[t].extend(new_cuts[t])
    
    # Calcul de z_inf (avec coupes)
    z_inf_new = 0
    x_inf = []
    for t in range(T):
        x_t, _ = solve_stage(t, None, cutting_planes[t])
        z_inf_new += c_vectors[t] @ x_t
        x_inf.append(x_t)
    
    history.append(("Backward", z_inf_new))
    
    print(f"\nIteration {it+1}:")
    print(f"z_sup = {z_sup_new:.2f}, z_inf = {z_inf_new:.2f}")
    print(f"Gap = {z_sup_new - z_inf_new:.2f}")
    
    if abs(z_sup_new - z_inf_new) < epsilon:
        break

# Résultats finaux
print("\nConvergence atteinte !")
print(f"z_sup final: {z_sup_new:.2f}")
print(f"z_inf final: {z_inf_new:.2f}")
print(f"Solution globale: {result_global.x}")