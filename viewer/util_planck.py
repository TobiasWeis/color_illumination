'''
approximation of planckian locus:
    Kim et. al:  US patent 7024034, Kim et al., "Color Temperature Conversion System and Method Using the Same", issued 2006-04-04

    Approximation formula: https://en.wikipedia.org/wiki/Planckian_locus
'''

import numpy as np

def calc_planck_locus():

    planck_points_cie = np.empty((0,2), np.float64)

    for T in np.arange(1667,25000, 1.):
        #x_c = 0.
        #y_c = 0.

        if T < 4000:
            x_c = -0.2661239*(10**9 / T**3) - 0.2343580*(10**6/T**2)+0.8776956*(10**3/T) + 0.179910
        else:
            x_c = -3.0258469*(10**9 / T**3) + 2.1070379*(10**6/T**2)+0.2226347*(10**3/T) + 0.240390

        if T < 2222:
            y_c = -1.1063814*(x_c**3) - 1.34811020*(x_c**2) + 2.18555832*x_c - 0.20219683
        elif T < 4000:
            y_c = -0.9549476*(x_c**3) - 1.37418593*(x_c**2) + 2.09137015*x_c - 0.16748867
        else:
            y_c = +3.0817580*(x_c**3) - 5.87338670*(x_c**2) + 3.75112997*x_c - 0.37001483

        planck_points_cie = np.append(planck_points_cie, np.array([[x_c,y_c]]), axis=0)
        #print "Appending for T=",T," : ", x_c, ", ", y_c

    print "Computed %d planck points" % (planck_points_cie.shape[0])

    return planck_points_cie


def calc_planck_locus_single_poly():
    '''
    gives planck locus points using only one singel polynom
    ( easier to calculate intersect later on )
    '''
    a = -2.7578
    b = 2.7318
    c = -0.2619

    x = np.linspace(0.2, 0.7, num=1000)

    # try the same with poly
    poly = np.polynomial.polynomial.Polynomial([c,b,a])

    planck_points_cie = np.empty((0,2), np.float64)

    for xp in x:
        planck_points_cie = np.append(planck_points_cie, np.array([[xp, poly(xp)]]), axis=0)

    return planck_points_cie

def calc_planck_locus_intersect(poly_a, poly_b, poly_c):
    '''
    takes the poly coeffs of a third deg. poly to calc intersect with
    planck locus blackbody curve in CIE space
    '''
    a = -2.7578
    b = 2.7318
    c = -0.2619

    roots = np.polynomial.polynomial.polyroots(np.array([c,b,a]) - [poly_c,poly_b,poly_a])

    ret = np.empty((0,2), np.float64)
    for root in roots:
        ret = np.append(ret, np.array([[root, poly_b*root + poly_c]]), axis=0)

    return ret
