#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:00:36 2023

@author: jbeckwith
"""

import numpy as np
import scipy as sp

class LF():
    def __init__(self):
        self = self
        return
    
    def BrownianTrans(self, DT, NAxes, TStep, NSteps):
        # BrownianTrans function
        # generates random translational motion of N coordinates 
        # using DT and TStep
        # ================INPUTS============= 
        # DT is translational diffusion coefficient
        # NAxes is an integer
        # TStep is time step relative to DT
        # NSteps is number of steps to simulate
        # ================OUTPUT============= 
        # coordinates are coordinates over time
        
        # get dimensionless translational diffusion coefficient (good explanation
        # in Allen, M. P.; Tildesley, D. J. Computer Simulation of Liquids,
        # 2nd ed.; Oxford University Press, 2017)
        sigmaT = np.sqrt(np.multiply(2., np.multiply(DT, TStep)))
        
        r0 = np.zeros([NAxes]) # initial position is 0s
        # generate random numbers for all steps
        rns = np.random.normal(loc=0, scale=sigmaT, size=(NAxes, NSteps-1))
        # make coordinates
        coordinates = np.vstack([r0, np.cumsum(rns, axis=1).T]).T

        return coordinates

    def BrownianRot(self, DR, TStep, NSteps):
        # BrownianRot function
        # generates random rotational motion
        # using DR and TStep
        # ================INPUTS============= 
        # DR is rotational diffusion coefficient
        # TStep is time step relative to DT
        # NSteps is number of steps to simulate
        # ================OUTPUT============= 
        # sph_coords are theta, phi over time

        # generate spherical coordinates using method of Hunter, G. L.;
        # Edmond, K. V.; Elsesser, M. T.; Weeks, E. R. 
        # Opt. Express 2011, 19 (18), 17189–17202.
        # Equations to convert Cartesian coordinates to Spherical coordinates
        # Rotation matrices
        r0 = np.zeros([3]) + np.sqrt(np.divide(1, 3.)) # initial position is sqrt(1/3)
        
        Rx = lambda alpha: np.array([[1, 0, 0], 
                                    [0, np.cos(alpha), -np.sin(alpha)], 
                                    [0, np.sin(alpha), np.cos(alpha)]])
        Ry = lambda beta: np.array([[np.cos(beta), 0, np.sin(beta)], 
                                    [0, 1, 0], 
                                    [-np.sin(beta), 0, np.cos(beta)]])
        Rz = lambda gamma: np.array([[np.cos(gamma), -np.sin(gamma), 0], 
                                    [np.sin(gamma), np.cos(gamma), 0], 
                                    [0, 0, 1]])
        
        sigmaT = np.sqrt(np.multiply(2., np.multiply(DR, TStep)))
        # equations to convert x y and z to theta and phi
        # see https://en.wikipedia.org/wiki/Spherical_coordinate_system
        theta = lambda x, y, z: np.arccos(np.divide(z, np.sqrt(x**2 + y**2 + z**2)))
        phi = lambda x, y: np.mod(np.arctan2(y, x), np.multiply(2., np.pi))
        
        # Simulate Rotational Diffusion
        coordinates = np.vstack([r0, np.zeros([NSteps-1, 3])]).T
        
        xyzdisp = np.random.normal(loc=0, scale=sigmaT, size=(3, NSteps-1))
        
        for j in np.arange(1, NSteps):
            r_prev = coordinates[:, j-1]
            coordinates[:, j] = np.matmul(np.matmul(np.matmul(Rx(xyzdisp[0, j-1]), Ry(xyzdisp[1, j-1])), Rz(xyzdisp[2, j-1])), r_prev)
            
        # Convert Cartesian coordinates to spherical coordinates
        sph_coords = np.array([theta(coordinates[0,:],coordinates[1,:],coordinates[2,:]), phi(coordinates[0,:],coordinates[1,:])]);
        return sph_coords