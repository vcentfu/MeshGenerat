import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import random 
import math


# Cas 1 : Carre

def grille_carre(c, f):
    
    """ int * int -> list[tuple(int)]
    
        Retourne les coordonnees des points formant la grille d'un carre de cote c et d'echantillonnage f. """
        
    grille = [(x * c / f, y * c / f) for y in range(f + 1) for x in range(f + 1)]
        
    return grille
    

def maillage_auto(c, f):
    
    """ int * int -> void
    
        Dessine un maillage (automatique) d'un carre de cote c et d'echantillonnage f. """
    
    grille = grille_carre(c, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    
    plt.triplot(x, y)
    
    return


def triangulation_regulier(c, f):
    
    """ int * int -> plt.tri.triangulation object
    
        Retourne un trigulation regulier d'un carre de cote c et d'echantillonnage f. """
        
    grille = grille_carre(c, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    connectivite = []
    
    nbp_line = f + 1
    
    for j in range(f):
        for i in range(f):
            connectivite.append([i + j * nbp_line, i + j * nbp_line + nbp_line, i + 1 + j * nbp_line + nbp_line])
            connectivite.append([i + j * nbp_line, i + 1 + j * nbp_line, i + 1 + j * nbp_line + nbp_line])
    
    return tri.Triangulation(x, y, connectivite)


def maillage_regulier(c, f):
    
    """ int * int -> void
    
        Dessine un maillage regulier d'un carre de cote c et d'echantillonnage f. """
        
    triang = triangulation_regulier(c, f)
    plt.triplot(triang)

    return


def triangulation_regulier_inv(c, f):
    
    """ int * int -> plt.tri.triangulation object
    
        Retourne un trigulation regulier inverse d'un carre de cote c et d'echantillonnage f. """
        
    grille = grille_carre(c, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    connectivite = []
    
    nbp_line = f + 1
    
    for j in range(f):
        for i in range(f):
            connectivite.append([i + j * nbp_line, i + j * nbp_line + nbp_line, i + 1 + j * nbp_line])
            connectivite.append([i + 1 + j * nbp_line + nbp_line, i + 1 + j * nbp_line, i + j * nbp_line + nbp_line])
    
    return tri.Triangulation(x, y, connectivite)
    

def maillage_regulier_inv(c, f):
    
    """ int * int -> void
    
        Dessine un maillage regulier inverse d'un carre de cote c et d'echantillonnage f. """
        
    triang = triangulation_regulier_inv(c, f)
    plt.triplot(triang)

    return


# Cas 2 : rectangle

def grille_rectangle(nb_extra, l, f) :
    
    """ int * int * int -> list[tuple(int)] 

        Retourne les coordonnees des points formant la grille d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre.  """      
    
    grille = [(x * l / f, y * l / f) for y in range(f + 1) for x in range(f + nb_extra + 1)]
        
    return grille


def maillage_auto_rec(nb_extra, l, f):
    
    """ int * int * int -> void
    
        Dessine un maillage (automatique) d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre. """
    
    grille = grille_rectangle(nb_extra, l, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    
    plt.triplot(x, y)
    
    return


def triangulation_regulier_rec(nb_extra, l, f):
    
    """ int * int *int  -> plt.tri.triangulation object
    
        Retourne un trigulation regulier d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre. """
        
    grille = grille_rectangle(nb_extra, l, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    connectivite = []
    
    nbp_line = f + nb_extra + 1
    
    for j in range(f):
        for i in range(f + nb_extra):
            connectivite.append([i + j * nbp_line, i + j * nbp_line + nbp_line, i + 1 + j * nbp_line + nbp_line])
            connectivite.append([i + j * nbp_line, i + 1 + j * nbp_line, i + 1 + j * nbp_line + nbp_line])
    
    return tri.Triangulation(x, y, connectivite)


def maillage_regulier_rec(nb_extra, l, f):
    
    """ int * int * int -> void
    
        Dessine un maillage regulier d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre. """
        
    triang = triangulation_regulier_rec(nb_extra, l, f)
    plt.triplot(triang)

    return
    
    
def triangulation_regulier_inv_rec(nb_extra, l, f):
    
    """ int * int * int -> plt.tri.triangulation object
    
        Retourne un trigulation regulier inverse d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre. """
        
    grille = grille_rectangle(nb_extra, l, f)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    connectivite = []
    
    nbp_line = f + nb_extra + 1
    
    for j in range(f):
        for i in range(f + nb_extra):
            connectivite.append([i + j * nbp_line, i + j * nbp_line + nbp_line, i + 1 + j * nbp_line])
            connectivite.append([i + 1 + j * nbp_line + nbp_line, i + 1 + j * nbp_line, i + j * nbp_line + nbp_line])
    
    return tri.Triangulation(x, y, connectivite)
    

def maillage_regulier_inv_rec(nb_extra, l, f):
    
    """ int * int * int -> void
    
        Dessine un maillage regulier inverse d'un rectangle de largeur l et de longueur nb_extra * (l / f) + l et d'echantillonnage f pour le sous carre. """
        
    triang = triangulation_regulier_inv_rec(nb_extra, l, f)
    plt.triplot(triang)

    return
    

# Cas non regulier
    
def grille_non_reg(L, l, c):
    
    """ int * int * int -> list[tuple(int)]
    
        Retourne une grille non reguliere contenant un sous carre. """
        
    nb_Lc = int(L / c)
    nb_lc = int(l / c)
    grille = []
    
    for y in range(nb_lc + 1):
        for x in range(nb_Lc + 1):
            grille.append((x * c, y * c))
        
        if nb_Lc != L / c :
            grille.append((L, y * c))
    
    if nb_lc != l / c :
        for x in range(nb_Lc + 1):
            grille.append((x * c, l))
    
        if nb_Lc != L / c :
            grille.append((L, l))
    
    return grille


def triangulation_non_reg(L, l, c):
    
    """ int * int * int -> plt.tri.Triangulation object

        Retourne une triangulation non regulier. """
        
    grille = grille_non_reg(L, l, c)
    
    x = [k for (k, l) in grille]
    y = [l for (k, l) in grille]
    connectivite = []
    
    nbp_Lline = int(L / c)
    nbp_lline = int(l / c)
    if nbp_Lline != L / c :
        nbp_L = nbp_Lline + 2
    else : 
        nbp_L = nbp_Lline + 1
        
    if nbp_lline != l / c :
        nbp_l = nbp_lline + 2
    else : 
        nbp_l = nbp_lline + 1
    
    for j in range(nbp_l - 1):
        for i in range(nbp_L - 1):
            connectivite.append([i + j * nbp_L, i + j * nbp_L + nbp_L, i + 1 + j * nbp_L + nbp_L])
            connectivite.append([i + j * nbp_L, i + 1 + j * nbp_L, i + 1 + j * nbp_L + nbp_L])
    
    return tri.Triangulation(x, y, connectivite)


def maillage_qcq(L, l, c):
    
    """ int * int * int -> void
    
        Retourne un maillage d'un rectangle. """
        
    t = triangulation_non_reg(L, l, c)
    plt.triplot(t)
    
    return


def pgcd(a, b):  
    
    """ int * int -> int
    
        Retourne le pgcd de a et b. """
    
    while a % b != 0 : 
        a, b = b, a % b 
    
    return b


def maillage_b(L, l):
    
    """ int * int -> void
    
        Retourne un maillage avec de bon proportion. """
        
    c = pgcd(L, l)
    
    if c != 1 :
        for i in range(2, c + 1):
            if c % i == 0:
                c = i
                break
    
    t = triangulation_non_reg(L, l, c)
    plt.triplot(t)
    
    return


###############################################################################
    

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


###############################################################################

# Version amelioree
    

def triangle_modifie(triangle, indice):
    
    """ list[int] * int -> list[list[int]]
    
        Retourne les triangles de Delaunay par propriete du sous-division. """
        
    l = [[triangle[0], triangle[1], indice], [triangle[0], triangle[2], indice], [triangle[1], triangle[2], indice]]
    
    return l 
    

def angle(p1, p2):
    
    """ list[int] * list[int] -> float
    
        Retourne l'angle relativement a p1 de p1p2 et l'abscisses. """
    
    lh = distance(p1, p2)
    
    if p2[0] > p1[0] :
        la = distance(p1, (p2[0], p1[1]))
        return math.acos(la / lh)
    
    la = distance(p1, (p1[0], p2[1]))    
    
    return math.pi / 2 + math.acos(la / lh)


def calcul_pivot(liste_points) :
    
    """ list[tuple(int)] -> tuple(int)
    
        Retourne le point pivot celle qui à la plus petite ordonnees. """
        
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
    
    """ list[tuple(int)] -> list[tuple(int)]

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
    
    """ list[tuple(int)] -> list[tuple(int)]
        hypothese : liste_points_trie est triee selon l'angle relativement au pivot
    
        Retourne l'enveloppe convexe de liste_points_trie. """
        
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
    
    """ tuple(int) ** 4 -> bool
    
        Retourne True si pt est dans le cercle circonscrit de p1, p2, p3, False sinon. """
    
    c = centre_circonscrit(p1, p2, p3)
    
    if distance(c, pt) <= distance(c, p1):
        return True
    
    return False


def intersection_droite_ptc(pt1, pt2, ptc):
    
    """ list[tuple(int)] * tuple(int) -> bool
    
        Retourne True si la demi-droite passant par [ptc parallele a l'axe des absicisses intersecte strictement pt1pt2, False sinon. """
        
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
    
    """ list[tuple(int)] * tuple(int) -> bool
    
        Retourne True si la demi-droite passant par ptc] parallele a l'axe des absicisses intersecte strictement pt1pt2, False sinon. """
        
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
    
    """ list[tuple(int)] * tuple(int) -> bool
    
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
    
    """
    
    """
    
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
    
    """
    
    """
    
    t = [random.randint(0, n)]
    
    while len(t) != 3:
        ind = random.randint(0, n)
        
        if ind not in t:
            t.append(ind)
            
    return t


def ec_plus_proche_point(ec, s_pt):
    
    """
    
    """
    
    dd = dict()
    ddmin = []
    
    for pt in s_pt:
        d_min = distance(pt, ec[0])
        
        for pec in ec[1:]:
            d = distance(pt, pec)
            
            if d < d_min:
                d_min = d
        
        dd[d_min] = pt
        ddmin.append(d_min)
        
    return dd[min(ddmin)]                
        


def triangulation_incrementale(nuage):
    
    """
    
    """
    
    tx = [k for (k, l) in nuage]
    ty = [l for (k, l) in nuage]
    start_triangle = get_start(len(nuage) - 1)
    triangles = [start_triangle]
    ln = [nuage[it] for it in start_triangle]
    ec = enveloppe_convexe(tri_angles(ln))
    ens = set(nuage) - set(ln)
    
    for k in ens:
        triangles_touches = []
        par_t = triangles.copy()
            
        for td in par_t:
            if dans_cercle(nuage[td[0]], nuage[td[1]], nuage[td[2]], k) :
                triangles_touches.append(td)
                triangles.remove(td)
            
        triangles_ajoutes = []
        triangle_int = []
        triangles_ext = []
        
        for tdt in triangles_touches:
            en_tdt = enveloppe_convexe(tri_angles([nuage[tdt[0]], nuage[tdt[1]], nuage[tdt[2]]]))
            
            if est_dans_enveloppe(en_tdt, k):
                triangle_int = tdt
            else:
                triangles_ext.append(tdt)
        
        triangles_a_int = []
        
        if triangle_int != []:
            for i in range(3):
                triangles_a_int.append([triangle_int[i], triangle_int[(i + 1) % 3], nuage.index(k)])
            
        for tde in triangles_ext:
            for i in range(3):
                if intersection(nuage[tde[i]], nuage[tde[(i + 1) % 3]], nuage[tde[(i + 2) % 3]], k):
                    iinter = tde[(i + 2) % 3]
                    break
                    
            for ktde in tde:
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
                        p_triangles_a_int = triangles_a_int.copy()
                        
                        for tai in p_triangles_a_int:
                            for itai in range(3):
                                if nuage.index(k) == tai[itai] or nuage.index(k) == tai[(itai + 1) % 3]:
                                    continue
                                if iinter == tai[itai] or iinter == tai[(itai + 1) % 3]:
                                    continue
                                if intersection(k, nuage[iinter], nuage[tai[itai]], nuage[tai[(itai + 1) % 3]]):
                                    triangles_a_int.remove(tai)
                                    break
                        
        for iec in range(len(ec)):
            flag = True
            p1 = ec[iec]
            p2 = ec[(iec + 1) % len(ec)]
            centre = centre_circonscrit(p1, p2, k)
            rayon = distance(centre, p1)
                
            for ptec in ln:
                if ptec == p1 or ptec == p2 :
                    continue
                if distance(ptec, centre) <= rayon:
                    flag = False
                    
            if flag :
                triangles_ajoutes.append([nuage.index(p1), nuage.index(p2), nuage.index(k)])
        
        triangles = triangles + triangles_ajoutes + triangles_a_int
        ln.append(k)
        ec = enveloppe_convexe(tri_angles(ln))
        ens = ens - {k}
    
    return (tri.Triangulation(tx, ty, triangles),len(triangles))
                
   
def affichage_nuage_Delaunay_inc(nuage) :
    
    """ list[tuple(float)] -> void
    
        Retourne le maillage avec les triangulations de Delaunay. """
        
    t = triangulation_incrementale(nuage)
    plt.tripcolor(t[0], facecolors = np.array([0] * t[1]))
    plt.triplot(t[0], color = "red")
    
    
    return         
            
            
            
                
        
        
    






    
    
    
    