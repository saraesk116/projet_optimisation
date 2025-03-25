
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


# Paramètres du problème



T = 3
n1, n2 = 2, 2  # Dimension des blocs
# Définitions des vecteurs et matrices
c1 = np.array([1, 2])  # Coefficients pour x1
c2 = np.array([3, 1])  # Coefficients pour x2
#c3 = np.array([4, 5])  # Coefficients pour x3
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
x_hats = {t: np.zeros(len(c_vectors[t])) for t in range(T)}
results = {}

""""
x1_chapeau= linprog(c1, A_ub=-A1, b_ub=-b1, bounds=(0,None), method='highs').x
x2_chapeau= linprog(c2, A_ub=-A2, b_ub=-b2, bounds=(0,None), method='highs').x
x3_chapeau= linprog(c3, A_ub=-A3, b_ub=-b3, bounds=(0,None), method='highs').x
print(f"x1_chapeau: {x1_chapeau}")
print(f"x2_chapeau: {x2_chapeau}")
print(f"x3_chapeau: {x3_chapeau}")"""

# Résolution du problème de programmation linéaire par linprog
# Contraintes globales combinées
# Construction de la matrice A pour T=3
A_l1 = np.concatenate((A1, np.zeros((A1.shape[0], A2.shape[1])), np.zeros((A1.shape[0], A3.shape[1]))), axis=1)
A_l2 = np.concatenate((E1, A2, np.zeros((A2.shape[0], A3.shape[1]))), axis=1)
A_l3 = np.concatenate((np.zeros((A3.shape[0], A1.shape[1])), E2, A3), axis=1)

A = np.concatenate((A_l1, A_l2, A_l3), axis=0)
b = np.concatenate((b1, b2, b3), axis=0)
c = np.concatenate((c1, c2, c3), axis=0)


def solve_stage_problem(c, A, b, cutting_planes):
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
    #2eme algo d'optim c2x2 utiliser solve_stage avec liste_vide ok
    #renvoyer que res dans la fonction
    #resoudre les pb à la main et afficher les matrices a chaque iteration 
    #revoir les reshape flatten
    #forme de cutting planes soit pi*x1 ou - ... revoir les signes 
    m, n = A.shape
    p = len(cutting_planes)
    xt_dim = len(c)
    if p == 0:  
        res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0,None), method='highs')
        return res
           
    else:
        
        
        A_cutting = np.vstack([ai for ai, _ in cutting_planes])  
        b_cutting = np.array([bi for _, bi in cutting_planes])
        
        A_aug = np.block([
            [A, np.zeros((m, 1))],   
            [A_cutting, np.ones((p, 1))]  
        ])
        b_aug = np.concatenate((b, b_cutting))
    
        c_aug = np.concatenate((c, [1]))  
        bounds = [(0, None) for _ in range(xt_dim)]
        bounds.append((None, None))

        res = linprog(c_aug, A_ub=-A_aug, b_ub=-b_aug, bounds = bounds, method='highs')
        return res 
        


# Initialisation
cutting_planes = [[] for i in range(T)] #liste de T listes vides qu'on va remplir les multiplicateurs de lagrange a chaque étage
z_sup = np.inf
z_min = -np.inf 
epsilon = 0.001
index=0





# Résoudre le problème de minimisation qui nous donne x1_hat
while z_sup - z_min > epsilon and index < 10:
    index += 1
    print(f"\n🔄 **Itération {index}**")

    # Étape (a): Forward Simulation (1 → T)
    for t in range(T):
        print(f"Iteration forward: {t+1}")
        if t==0:
            res = solve_stage_problem(c_vectors[t], A_matrices[t], b_vectors[t], cutting_planes[t])
            if res.success:
                x_hats[t] = res.x[:len(c_vectors[t])]
                
            else:
                raise ValueError(f"⚠️ Échec de l'optimisation à l'étape {t}")
        else: 
            rhs= b_vectors[t]-E_matrices[t-1]@ x_hats[t-1]
            res= solve_stage_problem(c_vectors[t], A_matrices[t], rhs, cutting_planes[t])
        if res.success:
            x_hats[t] = res.x[:len(c_vectors[t])]
            
        else:
            raise ValueError(f"⚠️ Échec de l'optimisation Forward à l'étape {t}")
        results[t] = res  # Stocker le résultat de chaque étape
        
        
    
    # Mise à jour de z_min après la dernière étape
    z_min = results[0].fun
    

    


        # Étape (b): Backward Recursion (T → 1)
    for t in range(T - 1, 0, -1):  # T-1, T-2, ..., 1
        #print(f"Iteration backward: {t +1}")
        nb_contraintes = A_matrices[t].shape[0]
    
    # Réutiliser x_hats[t-1] pour calculer rhs
        if t > 0:
            rhs_backward = b_vectors[t] - E_matrices[t-1] @ x_hats[t-1]
        else:
            rhs_backward = b_vectors[t]
    
        # Résoudre le sous-problème backward avec coupes actuelles
        res_backward = solve_stage_problem(c_vectors[t], A_matrices[t], rhs_backward, cutting_planes[t])
        if not res_backward.success:
            raise ValueError(f"⚠️ Échec Backward étape {t}")
    
        # Extraire les multiplicateurs de LAGRANGE BACKWARD
        res_py = -res_backward.ineqlin.marginals
        #ai = np.dot(res_py, np.vstack([E_matrices[t-1], 0]))
        print(f"Multiplicateurs BACKWARD (étape {t}): {res_py}")
        # Calculer la coupe pour l'étape t-1
        #print(res_py)
        print(nb_contraintes)
        print(b_vectors[t])
        if len(cutting_planes[t]) == 0:
            ai = np.dot(res_py[:nb_contraintes], E_matrices[t-1])
            bi= np.sum(res_py[:nb_contraintes]* b_vectors[t])
        else:
            ai = np.dot(res_py[:nb_contraintes], E_matrices[t-1])
            b_alpha = np.hstack([cut[1] for cut in cutting_planes[t]])
            bi = np.sum(res_py* np.concatenate((b_vectors[t],b_alpha)))  # rhs_backward, pas b_vectors[t]
        #ai = np.dot(res_py, E_matrices[t-1])
        
    
        # Ajouter la coupe à l'étape t-1
        cutting_planes[t-1].append((ai, bi))
        print(f"Coupes à l'étape {t-1}: {cutting_planes[t-1]}")
    
    # Mettre à jour x_hats[t-1] si nécessaire (optionnel)
    # x_hats[t-1] = res_backward.x[:len(c_vectors[t-1])  # Décommenter si besoin
   
    z_sup = sum(np.dot(c_vectors[t], x_hats[t]) for t in range(T))

        # 🔍 Affichage des résultats intermédiaires
    print(f"🟢 z_sup à l'itération {index} = {z_sup}")
    print(f"🟠 z_min à l'itération {index} = {z_min}")
    #print(f"🔵 x_hats à l'itération {index} = {x_hats}")


# ---------------------- Résultat Final ---------------------- #
print("\n✅ **Optimisation terminée.**")
print(f"✅ z_sup final: {z_sup}")
print(f"✅ z_inf final: {z_min}")
print(x_hats)
# Comparaison avec la méthode classique (résolution globale)
result = linprog(np.concatenate(c_vectors), A_ub=-A, b_ub=-b, method="highs")
print(f"\n🎯 **Résultat global avec la méthode classique:** {result.x}")