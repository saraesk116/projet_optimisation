import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
# Paramètres du problème
T = 3
n1, n2 = 2, 2  # Dimension des blocs

# Définitions des vecteurs et matrices
c1 = np.array([1, 2])  # Coefficients pour x1
c2 = np.array([3, 1])  # Coefficients pour x2
c3 = np.array([1, 1])

A1 = np.array([[2, -1], [1, 3]])  # Contraintes sur x1
b1 = np.array([1, 2])  # Second membre pour A1 x1 >= b1

E1 = np.array([[1, 0], [0, 1]])  # Contraintes pour E1 x1
A2 = np.array([[3, 2], [1, 4]])  # Contraintes sur A2 x2
b2 = np.array([4, 5])  # Second membre pour E1 x1 + A2 x2 >= b2

E2 = np.array([[2, 0], [0, 3]])  # Contraintes pour E2 x2
A3 = np.array([[1, 2], [2, 1]])  # Contraintes sur A3 x3
b3 = np.array([3, 6])  # Second membre pour E2 x2 + A3 x3 >= b3

c_vectors = [c1, c2, c3]
A_matrices = [A1, A2, A3]
b_vectors = [b1, b2, b3]
E_matrices = [E1, E2]

# Résolution globale pour vérification
A_l1 = np.concatenate((A1, np.zeros((A1.shape[0], 2)), np.zeros((A1.shape[0], 2))), axis=1)
A_l2 = np.concatenate((E1, A2, np.zeros((A2.shape[0], 2))), axis=1)
A_l3 = np.concatenate((np.zeros((A3.shape[0], 2)), E2, A3), axis=1)
A_global = np.concatenate((A_l1, A_l2, A_l3), axis=0)
b_global = np.concatenate((b1, b2, b3))
c_global = np.concatenate((c1, c2, c3))
result_global = linprog(c_global, A_ub=-A_global, b_ub=-b_global, method='highs')
print(f"Résultat global:\n{result_global.x}\n")

   

# Stocker les valeurs de z_sup et z_inf pour chaque itération
z_sup_values = []
z_inf_values = []

def solve_stage_problem(c, A, b, previous_vars=None, E_prev=None, cutting_planes=[]):
    if previous_vars is not None and E_prev is not None:
        b_adjusted = b - np.dot(E_prev, previous_vars)
    else:
        b_adjusted = b

    m, n = A.shape
    p = len(cutting_planes)
    if p == 0:
        res = linprog(c, A_ub=-A, b_ub=-b_adjusted, bounds=(0, None), method='highs')
    else:
        A_cut = np.array([ai for ai, _ in cutting_planes])
        b_cut = np.array([bi for _, bi in cutting_planes])
        A_aug = np.vstack([A, A_cut])
        b_aug = np.concatenate([b_adjusted, b_cut])
        res = linprog(c, A_ub=-A_aug, b_ub=-b_aug, bounds=(0, None), method='highs')
    
    if not res.success:
        raise ValueError(f"Échec de l'optimisation: {res.message}")
    return res.x

# Initialisation
cutting_planes = [[] for _ in range(T)]
x_hats = [None] * T
z_sup = np.inf
z_inf = -np.inf
epsilon = 0.001
max_iter = 10
iter_count = 0

while z_sup - z_inf > epsilon and iter_count < max_iter:
    iter_count += 1
    print(f"\n=== Itération {iter_count} ===")

    # Forward pass
    x_hats_forward = []
    for t in range(T):
        if t == 0:
            x = solve_stage_problem(c_vectors[t], A_matrices[t], b_vectors[t])
        else:
            E_prev = E_matrices[t-1]
            prev_x = x_hats_forward[t-1]
            x = solve_stage_problem(c_vectors[t], A_matrices[t], b_vectors[t], prev_x, E_prev, cutting_planes[t])
        x_hats_forward.append(x)
        print(f"x{t+1}_forward: {x}")

    z_sup = sum(np.dot(c, x) for c, x in zip(c_vectors, x_hats_forward))
    print(f"z_sup: {z_sup}")

    # Backward pass
    new_cutting_planes = [[] for _ in range(T)]
    for t in reversed(range(1, T)):
        E_prev = E_matrices[t-1]
        next_x = x_hats_forward[t]
        
        # Résoudre le problème dual pour obtenir les multiplicateurs
        A_t = A_matrices[t]
        b_t_adjusted = b_vectors[t] - np.dot(E_prev, x_hats_forward[t-1])
        res = linprog(np.zeros(A_t.shape[1]), A_ub=-A_t, b_ub=-b_t_adjusted, 
                      bounds=(0, None), method='highs')
        if not res.success:
            raise ValueError(f"Échec dual étape {t+1}: {res.message}")
        
        pi = res.ineqlin.marginals
        print(f"Pi pour étape {t+1}: {pi}")

        # Générer la coupe
        ai = np.dot(pi, E_prev)
        bi = np.dot(pi, b_vectors[t])
        new_cutting_planes[t-1].append((ai, bi))
        print(f"Coupure pour étape {t}: {ai}x + α >= {bi}")

    # Mise à jour des coupes
    for t in range(T):
        cutting_planes[t].extend(new_cutting_planes[t])

    # Calcul de z_inf
    z_inf = sum(res.fun for res in [linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs') 
                                   for c, A, b in zip(c_vectors, A_matrices, b_vectors)])
    print(f"z_inf: {z_inf}")
    


   

    # Stocker les valeurs de z_sup et z_inf
    z_sup_values.append(z_sup)
    z_inf_values.append(z_inf)
"""
# Tracer les valeurs de z_sup et z_inf
plt.plot(range(1, iter_count + 1), z_sup_values, label='z_sup')
plt.plot(range(1, iter_count + 1), z_inf_values, label='z_inf')
plt.xlabel('Itération')
plt.ylabel('Valeur')
plt.title('Évolution de z_sup et z_inf')
plt.legend()
plt.grid(True)
plt.show()
"""

print("\n=== Résultats finaux ===")
print(f"z_sup: {z_sup}")
print(f"z_inf: {z_inf}")
print(f"Solution globale: {result_global.x}")
