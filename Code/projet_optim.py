import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize


"""
minimisation basique avec python 
"""
# Paramètres du problème
c1= np.array([2,3])
c2= np.array([5,4])
#c2= np.array([5,4])




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


b1 = np.array([1, 2, 3])
b2 = np.array([9, 8, 2])
b = np.vstack((-b1, -b2)).flatten()





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
    if p == 0:  
        res = linprog(c, A_ub=-A, b_ub=-b, method='highs')
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

        res = linprog(c_aug, A_ub=-A_aug, b_ub=-b_aug , method='highs')
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
    #print(cutting_planes)
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
    print(ai)
    print(bi)
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
# print(f"z_sup final: {z_sup}")
#print(f"z_inf final: {z_min}")



result = linprog(c, A_ub=A, b_ub=b, method="highs")
print (f"resultat avec la méthode classique: {result.x}")
#print(linprog(c1,-A1,-b1,method="highs").x)
#print(linprog(c2,-A2,-b2+np.dot(E1,np.array([0,0.5])),method="highs"))
'''
c3 = np.concatenate((c1, [1])) 
A3 = np.block([
             [A1, np.zeros((3, 1))],   
             [np.array([7,8]), np.ones((1, 1))] 
         ])'''
#print(linprog(c3, A_ub=-A3, b_ub=-np.array([1,2,3,9]), method='highs').x)