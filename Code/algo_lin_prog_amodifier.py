import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


"""
minimisation basique avec python 
"""
# Paramètres du problème
c1= np.array([2,3])
c2= np.array([5,4])

c = np.hstack((c1.flatten(), c2.flatten()))  # Vecteur de coût de taille (4,1)
#vecteur de taille 4*1

# A1,A2 et E1 matrice 3x2 donc pas nécessairement carrée
A1 = np.array([[1, 2], 
               [3, 4], 
               [ 5, 6]])

A2= np.array([[5, 2], 
               [9, 3], 
               [ 8, 3]])

E1 = np.array([[7, 8], 
               [9, 1], 
               [ 2, 3]])

A = np.vstack((-A1 @ np.eye(2, 4), -(E1 @ np.eye(2, 4) + A2 @ np.eye(2, 4))))


b1= np.array([[1], [2], [3]])
b2= np.array([[9], [8], [2]])






# dans cutting planes on stocke les coupes ai =pi*E1 et bi= pi*b2

def solve_stage_problem(c1, A1, b1, E1, b2, Pi2_T, cutting_planes):
    """
    Résout le problème d'optimisation en fonction de la présence de contraintes de cutting planes.

    Paramètres :
    - c1 : Vecteur coût pour x1
    - A1, b1 : Contraintes A1 * x1 >= b1
    - E1, b2 : Paramètres pour le cas complexe
    - Pi2_T : Matrice de projection
    - cutting_planes : Liste des contraintes supplémentaires (ai, bi) mises à jour

    Retour :
    - Solution optimisée (x1, alpha)
    """
    #2eme algo d'optim c2x2 utiliser solve_stage avec liste_vide
    #renvoyer que res dans la fonction
    #resoudre les pb à la main et afficher les matrices a chaque iteration 
    #revoir les reshape flatten
    #forme de cutting planes soit pi*x1 ou - ... revoir les signes 
    
    if len(cutting_planes)==0:  # Si la liste est vide, résoudre le problème simple
        res = linprog(c1, A_ub=-A1, b_ub=-b1, method='highs')
        if res.success:
            return res.x, 0, res.fun  # Alpha = 0 car aucun cutting plane
        else:
            raise ValueError("Problème d'optimisation non résolu (cas simple).")
    
    else:
        # Cas avec les contraintes de cutting planes
        m, n = A1.shape
        p = len(cutting_planes)  # Nombre de contraintes ajoutées
        
        # Matrices supplémentaires à partir de cutting_planes
        A_cutting = np.vstack([ai for ai, _ in cutting_planes])  # Matrice des a_i
        b_cutting = np.array([bi for _, bi in cutting_planes]).reshape(-1, 1)  # Vecteur des b_i

        # Construction de la matrice augmentée (voir image)
        A_aug = np.block([
            [A1, np.zeros((m, 1))],   # Contraintes de base
            [-A_cutting, np.ones((p, 1))]  # Contraintes des cutting planes
        ])
        
        # Vecteur des contraintes augmentées
        b_aug = np.vstack((b1.reshape(-1, 1), b_cutting))

        # Vecteur coût avec alpha
        c_aug = np.concatenate((c1, [1]))  # Étendre c1 avec le coût pour alpha

        # Résolution avec linprog
        res = linprog(c_aug, A_ub=-A_aug, b_ub=-b_aug.flatten(), method='highs')
        
        if res.success:
            return res.x[:-1], res.x[-1], res.fun  # Retourner x1 et alpha
        else:
            raise ValueError("Problème d'optimisation non résolu (cas complexe).")
        



# Initialisation
cutting_planes = []
z_sup = np.inf
z_inf = 0
epsilon = 0.001
index=0

# Multiplicateurs de Lagrange (initiaux)
py = np.zeros((3, 1))

# Résoudre le problème de minimisation qui nous donne x1_hat
while z_sup[0] - z_inf > epsilon and index < 10:
    index = index + 1
    print(cutting_planes)
    # Étape (a): Résolution du premier sous-problème
    res_x1,  alpha, z_min= solve_stage_problem(c1, A1, b1, E1, b2, py.T, cutting_planes)      # z_min est le cout du probleme 1
    #le reshape transpose  le vecteur ligne en vecteur colonne
    x1_hat = res_x1.reshape(-1, 1)

    # Étape (b): Résolution du deuxième sous-problème
    res_x2 = linprog(c2, A_ub=-A2, b_ub=(-b2 + np.dot(E1, x1_hat)).flatten(), method='highs')
    if not res_x2.success:
        print("Problème lors de la résolution du deuxième sous-problème.")
        break
    x2_hat = res_x2.x.reshape(-1, 1)
    res_py= -res_x2.ineqlin.marginals

    print(res_x2)
    print(f"res_x2  : {res_x2},")

    # res_py correspond à la valeur des multiplicateurs de Lagrange associés aux contraintes de x2_hat
    # on stocke le couple (ai, bi)= (res_py.transpose*E1,res_py.transpose*E2) dans cutting_planes
     # Calcul du nouveau cutting plane
    ai = np.dot(res_py, E1)
    bi = np.dot(res_py, b2)

    # Ajout aux cutting planes
    cutting_planes.append((ai, bi))

    # Étape (c): Calcul du coût de la fonction
    #alpha = np.dot(c2, x2_hat)

    py = res_py.reshape(-1, 1)

    # Étape (e): Calcul de z_sup
    z_sup = np.dot(c1, x1_hat) + np.dot(c2,x2_hat)
    print(f"z_sup à l'iteration  : {z_sup},{index}")
    print(f"z_min à l'iteration   : {z_min},{index}")

    # Étape (f): Mise à jour de alpha_hat (borne inférieure)
    # On utilise np.min car alpha_hat est une borne inférieure, donc on prend le max des lower bounds
    

    #print(f"Nouveau cutting plane ajouté: ai = {ai}, bi = {bi}")
    #print(f" z_sup: {z_sup}, z_inf: {z_inf}")

# Résultat final
print("Optimisation terminée.")
print(f"z_sup final: {z_sup}")
print(f"z_inf final: {z_inf}")
