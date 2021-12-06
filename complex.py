import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import random 
import math
import time

def nuage_pts(a, nb):
    
    """ int * int -> list[tuple(float)]
    
        Retourne un nuage de points aleatoire de taille nb ayant des coordonnees comprises entre 0 et a. """
        
    l = [(random.uniform(0, a), random.uniform(0, a)) for k in range(nb)]
    
    return l


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


def affichage_nuage_Delaunay(nuage) :
    
    """ list[tuple(float)] -> void
    
        Retourne le maillage avec les triangulations de Delaunay. """
        
    t = triangulation_Delaunay(nuage)
    plt.tripcolor(t[0], facecolors = np.array([0] * t[1]))
    plt.triplot(t[0], color = "red")
    
    
    return


###############################################################################
# Travail apres la seance 3
# Version amelioree
    

def angle(p1, p2):
    
    """ tuple(float) * tuple(float) -> float
    
        Retourne l'angle entre l'axe des abscisses et [p1p2]. """
    
    lh = distance(p1, p2)
    
    if p2[0] > p1[0] :
        la = distance(p1, (p2[0], p1[1]))
        return math.acos(la / lh)
    
    la = distance(p1, (p1[0], p2[1]))    
    
    return math.pi / 2 + math.acos(la / lh)


def calcul_pivot(liste_points) :
    
    """ list[tuple(float)] -> tuple(float)
    
        Retourne le point pivot dans l'algorthrime de Graham (celle qui à la plus petite ordonnees). """
        
    lo = [k[1] for k in liste_points]
    v_m = min(lo)
    lmi = [i for i in range(len(liste_points)) if liste_points[i][1] == v_m]
    
    if len(lmi) == 1 :
        return liste_points[lmi[0]]
    
    lp = [liste_points[l] for l in lmi]
    v_a_m = lp[0][0]
    res = lp[0]
    
    for h in lp[1:]:
        if h[0] < v_a_m :
            v_a_m = h[0]
            res = h
            
    return res
            
        
def tri_angles(liste_points) :
    
    """ list[tuple(float)] -> list[tuple(float)]

        Retourne la liste de points tries selon l'angle par rapport au point pivot. """
    
    lpc = liste_points.copy()   
    pivot = calcul_pivot(liste_points)
    l = [pivot]
    lpc.remove(pivot)
    d = {angle(pivot, pt):pt for pt in lpc}
    s = sorted(d)
    l = l + [d[k] for k in s]
    
    return l


def enveloppe_convexe(liste_points_trie):
    
    """ list[tuple(float)] -> list[tuple(float)]
        hypothese : liste_points_trie est triee selon l'angle relativement au pivot
    
        Retourne l'enveloppe convexe de liste_points_trie par l'algorthrime de Graham. """
        
    ec = [liste_points_trie[0], liste_points_trie[1]]
    
    for pt in liste_points_trie[2:]:
        (x1, y1) = ec[-2]
        (x2, y2) = ec[-1]
        (x3, y3) = pt
        pv = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        
        if pv < 0:
            ec.remove(ec[-1])
            n = len(ec)
            for i in range(n - 2):
                (x1, y1) = ec[-2]
                (x2, y2) = ec[-1]
                pv = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                
                if pv < 0:
                    ec.remove(ec[-1])
                
        
        ec.append(pt)
        
    return ec


def dans_cercle(p1, p2, p3, pt) :
    
    """ tuple(float) ** 4 -> bool
    
        Retourne True si pt est dans le cercle circonscrit de p1, p2, p3, False sinon. """
    
    c = centre_circonscrit(p1, p2, p3)
    
    if distance(c, pt) <= distance(c, p1):
        return True
    
    return False


def intersection_droite_ptc(pt1, pt2, ptc):
    
    """ tuple(float) ** 3 -> bool
    
        Retourne True si la demi-droite passant par [ptc parallele a l'axe des absicisses intersecte strictement [pt1pt2], False sinon. """
        
    (x1, y1) = pt1
    (x2, y2) = pt2
    (xc, yc) = ptc
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x_i = (yc - b) / a
    
    if abs((y1 + y2) / 2 - yc) > abs((y1 + y2) / 2 - y2):
        return (False, False)
    
    if x_i > max(x1, x2):
        return (False, False)
    
    if xc > x_i :
        return (False, False)
    
    if abs(y1 - yc) == y2 or xc == x_i:
        return (True, True)
    
    return (True, False)
    

def intersection_gauche_ptc(pt1, pt2, ptc):
    
    """ tuple(float) ** 3 -> bool
    
        Retourne True si la demi-droite passant par ptc] parallele a l'axe des absicisses intersecte strictement [pt1pt2], False sinon. """
        
    (x1, y1) = pt1
    (x2, y2) = pt2
    (xc, yc) = ptc
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x_i = (yc - b) / a
    
    if abs((y1 + y2) / 2 - yc) > abs((y1 + y2) / 2 - y2):
        return (False, False)
    
    if x_i < min(x1, x2):
        return (False, False)
    
    if xc < x_i :
        return (False, False)
    
    if abs(y1 - yc) == y2 or xc == x_i:
        return (True, True)
    
    return (True, False)
    

def est_dans_enveloppe(ec, pt):
    
    """ list[tuple(float)] * tuple(float) -> bool
    
        Retourne True si pt est dans l'enveloppement convexe, False sinon. """
    
    nb = 0
    flag = False
    
    for i in range(len(ec)):
        if intersection_droite_ptc(ec[i], ec[(i + 1) % len(ec)], pt) == (True, False):
            nb = nb + 1
        if intersection_droite_ptc(ec[i], ec[(i + 1) % len(ec)], pt) == (True, True):
            flag = True
    
    if nb == 0 and flag:
        for i in range(len(ec)):
            if intersection_gauche_ptc(ec[i], ec[(i + 1) % len(ec)], pt) == (True, False):
                nb = nb + 1
    
    if nb % 2 == 0:
        return False
        
    return True


def intersection(p1, p2, p3, p4) :
    
    """ tuple(float) ** 4 -> bool
    
        Retourne True si [p1p2] intersecte [p3p4], False sinon. """
    
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = p3
    (x4, y4) = p4
    
    a1 = (y1 - y2) / (x1 - x2)
    a2 = (y3 - y4) / (x3 - x4)
    b1 = y1 - a1 * x1
    b2 = y3 - a2 * x3
    
    if a1 * b2 - a2 * b1 == 0:
        return False
    
    res_x = (b2 - b1) / (a1 - a2)
    res_y = a1 * res_x + b1

    if abs((x1 + x2) / 2 - res_x) > abs((x1 + x2) / 2 - x2) or abs((y1 + y2) / 2 - res_y) > abs((y1 + y2) / 2 - y2):
        return False
    
    if abs((x3 + x4) / 2 - res_x) > abs((x3 + x4) / 2 - x4) or abs((y3 + y4) / 2 - res_y) > abs((y3 + y4) / 2 - y4):
        return False
    
    return True


def get_start(n):
    
    """ int -> list[int]
    
        Retourne un triangle aléatoire pour n + 1 points. """
    
    t = [random.randint(0, n)]
    
    while len(t) != 3:
        ind = random.randint(0, n)
        
        if ind not in t:
            t.append(ind)
            
    return t


def triangulation_incrementale(nuage):
    
    """ list[tuple(float)] -> plt.tri.Triangulation object
    
        Retourne une triangulation de Delaunay d'un nuage de points. """
    
    tx = [k for (k, l) in nuage] # pour triangulation object 
    ty = [l for (k, l) in nuage] # pour triangulation object
    start_triangle = get_start(len(nuage) - 1) # triangle de depart
    triangles = [start_triangle] 
    ln = [nuage[it] for it in start_triangle]
    ec = enveloppe_convexe(tri_angles(ln)) # l'enveloppe convexe du maillage
    ens = set(nuage) - set(ln)
    
    for k in ens:
        triangles_touches = [] # liste des triangles ou son cercle circonscrit atteint k
        copy_t = triangles.copy()
            
        for td in copy_t:
            if dans_cercle(nuage[td[0]], nuage[td[1]], nuage[td[2]], k) :
                triangles_touches.append(td)
                triangles.remove(td) # si son cercle atteint k, on retire ce triangle
            
        triangles_ajoutes = []
        triangle_int = [] # le triangle interieur ou k est dedans 
        triangles_ext = [] # les autres triangles ou k n'est pas dedans mais son cercle circonscrit l'atteint
        
        for tdt in triangles_touches:
            en_tdt = enveloppe_convexe(tri_angles([nuage[tdt[0]], nuage[tdt[1]], nuage[tdt[2]]])) # enveloppe convexe des triangles touches
            
            if est_dans_enveloppe(en_tdt, k): # si dedans on modifie triangle_int et on sait qu'il y en a qu'un seul triangle int
                triangle_int = tdt
            else:
                triangles_ext.append(tdt)
        
        triangles_a_int = [] # les 3 nouveaux triangles issus de triangle int par la propriété de Delaunay
        
        if triangle_int != []:
            for i in range(3):
                triangles_a_int.append([triangle_int[i], triangle_int[(i + 1) % 3], nuage.index(k)])
            
        for tde in triangles_ext: # si les triangles sont a l'exterieur, on n'en construit que 2 nouveaux triangles pour chaque triangle ext
            for i in range(3):
                if intersection(nuage[tde[i]], nuage[tde[(i + 1) % 3]], nuage[tde[(i + 2) % 3]], k): # on cherche le point n° iinter dans chaque triangle ext tel que [nuage[iinter]k] coupe un des segement de triangle ext 
                    iinter = tde[(i + 2) % 3]
                    break
                    
            for ktde in tde:   # on verifie la condition de Delaunay pour les 2 nouveaux triangles
                if ktde != iinter:
                    flag = True
                    p1 = nuage[ktde]
                    p2 = nuage[iinter]
                    centre = centre_circonscrit(p1, p2, k)
                    rayon = distance(centre, p1)
                    
                    for ptln in ln:
                        if ptln == p1 or ptln == p2 :
                            continue
                        if distance(ptln, centre) <= rayon:
                            flag = False
                    
                    if flag:
                        triangles_ajoutes.append([ktde, iinter, nuage.index(k)])
                        copy_triangles_a_int = triangles_a_int.copy()
                        
                        for tai in copy_triangles_a_int: # on regarde si les 2 nouveaux triangles ne croisent pas les 3 nouveaux triangles interieurs 
                            for itai in range(3):
                                if nuage.index(k) == tai[itai] or nuage.index(k) == tai[(itai + 1) % 3]:
                                    continue
                                if iinter == tai[itai] or iinter == tai[(itai + 1) % 3]:
                                    continue
                                if intersection(k, nuage[iinter], nuage[tai[itai]], nuage[tai[(itai + 1) % 3]]):
                                    triangles_a_int.remove(tai) # s'il y a croisement on supprime
                                    break
                        
        for iec in range(len(ec)): # dans le cas ou k se trouve a l'exterieur du maillage courant 
            flag2 = True           # on doit verifier que les points 2 a 2 du enveloppe convexe du maillage courant forme un triangle avec k ou non 
            p1 = ec[iec]
            p2 = ec[(iec + 1) % len(ec)]
            centre = centre_circonscrit(p1, p2, k)
            rayon = distance(centre, p1)
                
            for ptec in ln:
                if ptec == p1 or ptec == p2 :
                    continue
                if distance(ptec, centre) <= rayon:
                    flag2 = False
                    
            if flag2 :
                triangles_ajoutes.append([nuage.index(p1), nuage.index(p2), nuage.index(k)])
        
        triangles = triangles + triangles_ajoutes + triangles_a_int
        ln.append(k)
        ec = enveloppe_convexe(tri_angles(ln)) # on reactualise l'enveloppe convexe du maillage
        ens = ens - {k}
    
    return (tri.Triangulation(tx, ty, triangles),len(triangles))
                
   
def affichage_nuage_Delaunay_inc(nuage) :
    
    """ list[tuple(float)] -> void
    
        Retourne le maillage avec les triangulations de Delaunay. """
        
    t = triangulation_incrementale(nuage)
    plt.tripcolor(t[0], facecolors = np.array([0] * t[1]))
    plt.triplot(t[0], color = "red")
    
    
    return 


x = np.arange(20, 500, 20)
y = []
for xi in x:
    print(xi)
    nuage = nuage_pts_d(40, xi, 0.2)
    start = time.process_time()
    triangulation_Delaunay(nuage)
    end = time.process_time()
    y.append(end - start)

plt.figure()
plt.title('courbe du temps d\'execution')
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.xlabel('nombre de points')
plt.ylabel('temps en secondes')

plt.subplot(2, 1, 2)
logx = np.log(x)
logy = np.log(y)
pente = (logy[-1] - logy[0]) / (logx[-1] - logx[0])
plt.plot(logx, logy, label='pente=%1.3f'%pente)
plt.xlabel('log(nombre de points)')
plt.ylabel('log(temps)')
plt.tight_layout()
plt.legend()