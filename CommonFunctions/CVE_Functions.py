#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:00:36 2023

@author: jbeckwith
"""

import numpy as np

class CVE():
    def __init__(self):
        self = self
        return
    
    def Eq14(self, x, t, R=1./6):
        """ Eq14 function
        # takes positions and uses equation 14 from
        # Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        # Phys. Rev. E 2014, 89 (2), 022726. 
        # https://doi.org/10.1103/PhysRevE.89.022726.
        # to estimate diffusion coefficient and
        # localisation precision
        # ================INPUTS============= 
        # x is 1D positions
        # t is time
        # R is R parameter (see Equation 5 of paper)
        # ================OUTPUT============= 
        # D is diffusion coefficient estimate
        # sigma is localisation precision
        # varD is variance of D """
        diffX = np.diff(x)
        mult = np.mean(np.multiply(diffX[:-1], diffX[1:]))
        deltaX_sqr = np.mean(np.square(diffX))
        deltat = np.mean(np.unique(np.diff(t)))
        D = np.add(np.divide(deltaX_sqr, np.multiply(2, deltat)), np.divide(mult, deltat))
        
        sigma = np.sqrt(np.multiply(R, deltaX_sqr) + np.multiply((2*R - 1), mult))
        
        epsilon = np.subtract(np.divide(np.square(sigma), np.multiply(D, deltat)), np.multiply(2, R))
        N = len(x)
        varD = np.multiply(np.square(D), (((6 + 4*epsilon + 2*np.square(epsilon))/N) + ((4*np.square(1+epsilon))/np.square(N))))
        return D, sigma, varD
    
    def Eq16(self, x, sigma, t, R=1./6):
        """ Eq16 function
        # takes positions and uses equation 16 from
        # Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        # Phys. Rev. E 2014, 89 (2), 022726. 
        # https://doi.org/10.1103/PhysRevE.89.022726.
        # to estimate diffusion coefficient and
        # localisation precision
        # ================INPUTS============= 
        # x is 1D positions
        # sigma is precision of estimations of x
        # t is time
        # R is R parameter (see Equation 5 of paper)
        # ================OUTPUT============= 
        # D is diffusion coefficient estimate
        # varD is variance of D """
        diffX = np.diff(x)
        sigma_squared = np.square(np.mean((sigma)))
        deltaX_sqr = np.mean(np.square(diffX))
        deltat = np.mean(np.unique(np.diff(t)))
        D = np.divide(np.subtract(deltaX_sqr, np.multiply(2., sigma_squared)), np.multiply(np.multiply(2, np.subtract(1., np.multiply(2., R))), deltat))
        varsigma = np.var(np.square(sigma))
        epsilon = np.subtract(np.divide(np.square(sigma), np.multiply(D, deltat)), np.multiply(2, R))
        N = len(x)
        varD = np.add(np.divide(np.multiply(np.square(D), (2 + 4*epsilon + 3*np.square(epsilon))), N*np.square(1 - 2*R)), np.divide(varsigma, (np.square(deltat)*np.square(1 - 2*R))))
        
        return D, varD
    
    def Eq10(self, D, deltaT, sigma, k, N, R=1./6):
        """ Eq10 function
        # calculates theoretical form of power spectrum from
        # Vestergaard, C. L.; Blainey, P. C.; Flyvbjerg, H
        # Phys. Rev. E 2014, 89 (2), 022726. 
        # https://doi.org/10.1103/PhysRevE.89.022726.
        # to estimate if particle is undergoing diffusive motion or not
        # ================INPUTS============= 
        # D is diffusion coefficient
        # deltaT is time difference between displacements
        # sigma is precision
        # k is modes of discrete sine transform
        # N is number of data points
        # R is R parameter (see Equation 5 of paper)
        # ================OUTPUT============= 
        # theoretical form of the power spectrum """
        term1 = np.multiply(D, np.square(deltaT))
        term2_1 = np.multiply(np.square(sigma), deltaT)
        term2_2 = np.multiply(2., np.multiply(D, np.multiply(R, np.square(deltaT))))
        term3 = np.subtract(1., np.cos(np.divide(np.multiply(np.pi, k), np.add(N, 1.))))
        term2 = np.multiply(np.multiply(2., np.subtract(term2_1, term2_2)), term3)
        return np.add(term1, term2)