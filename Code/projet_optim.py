import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize

"""
minimisation basique avec python 
"""
# Paramètres du problème

A1 = np.array([[2, 2], [2, -1], [2, 1]])  # Contraintes plus variées en R^2

b1 = np.array([9, 2, 3])
c1 = np.array([4, 1])

E1 = np.array([[1, 1], [2, -1], [-1, 0]])

A2 = np.array([[3, 2], [-1, 2], [1, -1]])  # Contraintes pour la 2ème étape
b2 = np.array([1, 3, 1])
c2 = np.array([2, 2])

"""
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
"""



# Résolution du problème de programmation linéaire par linprog
# Contraintes globales combinées
A_l1 = np.concatenate((A1,np.zeros((A1.shape[0],A2.shape[1]))),axis=1)
A_l2 = np.concatenate((E1,A2),axis=1)
A = np.concatenate((A_l1,A_l2),axis=0)
b = np.concatenate([b1, b2],axis=0)
# Fusionner les coefficients de la fonction objectif
c = np.concatenate((c1, c2),axis=0)

#print(A)




# dans cutting planes on stocke les coupes ai =pi*E1 et bi= pi*b2

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
        #print(A1,c1,b1)
           
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
cutting_planes = []
z_sup = np.inf
z_min = -np.inf 
epsilon = 0.001
index=0




# Résoudre le problème de minimisation qui nous donne x1_hat
while z_sup - z_min > epsilon and index < 10:
    index = index + 1
    

    # Étape (a): Résolution du premier sous-problème
    res1= solve_stage_problem(c1, A1, b1, cutting_planes)     
    if not res1.success:
        raise ValueError("Échec de l'optimisation du premier sous-problème")

    if len(cutting_planes) > 0:
        x1_hat = res1.x[:-1]  # Exclure alpha
        alpha = res1.x[-1]
    else:
        x1_hat = res1.x
        alpha = 0
    z_min = res1.fun

    # Étape (b): Résolution du deuxième sous-problème
    rhs = b2 - E1 @ x1_hat
    res2 = solve_stage_problem(c2, A2, rhs, [])

    if not res2.success:
        raise ValueError("Échec de l'optimisation du deuxième sous-problème")

    # Vérification que les multiplicateurs sont non négatifs
    
    x2_hat = res2.x[:len(c2)]  # Prend seulement les éléments correspondant à c2
    res_py= np.abs(res2.ineqlin.marginals)
    
    # res_py correspond à la valeur des multiplicateurs de Lagrange associés aux contraintes de x2_hat
    # on stocke le couple (ai, bi)= (res_py.transpose*E1,res_py.transpose*E2) dans cutting_planes
     # Calcul du nouveau cutting plane
    ai = np.dot(res_py, E1)
    bi = np.dot(res_py, b2)


    # Ajout aux cutting planes
    cutting_planes.append((ai, bi))


    # Étape (c): Calcul de z_sup
    print(f"res_py: {res_py}")
    z_sup = np.dot(c1, x1_hat) + np. dot(c2,x2_hat)
    
    print(f"z_sup à l'iteration {index} est égal à  {z_sup}")
    print(f"z_min à l'iteration {index}est égal à {z_min}")
    print (f"x1_hat: {x1_hat}")
    print (f"x2_hat: {x2_hat}")
    

# Résultat final

print("Optimisation terminée.")
print(f"z_sup final: {z_sup}")
print(f"z_inf final: {z_min}")


result = linprog(c, A_ub=-A, b_ub=-b, method="highs")
print (f"resultat avec la méthode classique: {result.x}")

