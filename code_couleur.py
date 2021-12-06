import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import random 
import math

def nuage_pts_d(a, nb, d):
    
    """ int * int -> list[tuple(float)]
    
        Retourne un nuage de points aleatoire de taille nb ayant des coordonnees comprises entre 0 et a distance de d. """
        
    l = [(random.uniform(0, a), random.uniform(0, a))]
    
    for k in range(nb - 1):
        trouve = False
        
        while not trouve :
            p_c = (random.uniform(0, a), random.uniform(0, a))
            trouve = True
            
            for p in l :
                if distance(p_c, p) <= d :
                    trouve = False
                    break
            
        l.append(p_c)
    
    return l 
                

def affichage_nuage(nuage) :
    
    """ list[tuple(float)] -> void
    
        Dessine le nuage de point. """
        
    x = [k for (k, l) in nuage]
    y = [l for (k, l) in nuage]
    
    plt.scatter(x, y)
    
    return


def combinaisons(nb):
    
    """ int -> list[list[int]]
    
        Retourne les combinaisons possibles de 3 indices dans un ensemble de nb indices. """
        
    cb = [[i, j, k] for i in range(nb) for j in range (i + 1, nb) for k in range(j + 1, nb)]
    
    return cb


def equation_mediatrice(p1, p2):
    
    """ tuple(float) * tuple(float) -> tuple(float)
    
        Retourne les coefficients de la combinaison lineaire de l'équation de la médiatrice de p1 et p2. """
     
    a = 2 * (p2[0] - p1[0])
    b = 2 * (p2[1] - p1[1])
    c = p1[0] ** 2 + p1[1] ** 2 - p2[0] ** 2 - p2[1] ** 2
    
    return (a, b, c)


def centre_circonscrit(p1, p2, p3):
    
    """ tuple(float) * tuple(float) * tuple(float) -> tuple(float)

        Retourne le centre du cercle circonscrit de p1p2 et p1p3. """
        
    (a1, b1, c1) = equation_mediatrice(p1, p2)
    (a2, b2, c2) = equation_mediatrice(p1, p3)
    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([[-c1], [-c2]])
    X = np.linalg.inv(A).dot(B)
    
    return (X[0, 0], X[1, 0])


def distance(p1, p2):
    
    """ tuple(int) * tuple(int) -> float
    
        Retourne la distance euclidienne entre les points p1 et p2. """
        
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def triangulation_Delaunay(nuage):
    
    """ list[tuple(float)] -> plt.tri.Triangulation object
    
        Retourne une triangulation de Delaunay d'un nuage de points. """
        
    tx = [k for (k, l) in nuage]
    ty = [l for (k, l) in nuage]
    
    l_triangles_ini = combinaisons(len(nuage))
    res_triang = []
    
    for triangle in l_triangles_ini :
        flag = True
        p1 = nuage[triangle[0]]
        p2 = nuage[triangle[1]]
        p3 = nuage[triangle[2]]
        centre = centre_circonscrit(p1, p2, p3)
        rayon = distance(centre, p1)
        
        for ind in range(len(nuage)) :
            if ind not in triangle :
                pt = nuage[ind]
                distance_t = distance(centre, pt)
                
                if distance_t <= rayon :
                    flag = False
                    break
                
        if flag :
            res_triang.append(triangle)
    
    n = len(res_triang)
    
    return (tri.Triangulation(tx, ty, res_triang), n)


def affichage_nuage_Delaunay_edge(nuage) :
    """ list[tuple(float)] -> void
    
        Retourne le maillage avec les triangulations de Delaunay. """
        
    t = triangulation_Delaunay(nuage)
    plt.triplot(t[0], color = "red")
    
    return

def affichage_nuage_Delaunay(nuage) :
    
    """ list[tuple(float)] -> void
    
        Retourne le maillage avec les triangulations de Delaunay. """
        
    t = triangulation_Delaunay(nuage)
    plt.tripcolor(t[0], facecolors = np.array([0] * t[1]))
    plt.triplot(t[0], color = "red")
    
    
    return