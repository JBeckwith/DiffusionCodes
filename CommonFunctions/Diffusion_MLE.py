#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 08:54:52 2020

@author: jbeckwith
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import numpy as np
from lmfit import Model
from pathos.pools import ThreadPool
from copy import deepcopy


class TrackAnalysis():
    def __init__(self, track, params):
        self.track = track
        self.params = params
        
    # Calculate apparent and corrected diffusion coefficients
    # using *overlapping* time lags
    @staticmethod
    #@jit(nopython=True, parallel=True)
    def MLEloop(X, Y, Z, lags, m, KFC, TimeStep):
        # Preallocate arrays for apparent and corrected diffusion
        # coefficient MLEs, as well as number of time lags of a
        # given length
        DappX = np.zeros(lags-1)
        DappY = np.zeros(lags-1)
        DappZ = np.zeros(lags-1)
        DcorrX = np.zeros(lags-1)
        DcorrY = np.zeros(lags-1)
        DcorrZ = np.zeros(lags-1)
        N = np.zeros(lags-1)
        dns = np.arange(1, lags)
        # Naive MLE of diffusion coefficient needs to be corrected
        # by this function as per the following reference: Xu, C. S.; Cang, H.; Montiel, D.; Yang, H.. J. Phys. Chem. C 2007, 111 (1), 32–35.
        # equation is [2 − 2(1 − λ )^(1 + m) − 2λ − 2m λ + m λ 2 ] /[ m(−2 + λ ) λ, lambda is Kalman factor
        mfactor = np.multiply(m, dns)
        cfn2 = np.multiply(2, np.power(np.subtract(1, KFC), np.add(1, mfactor)))
        cfn3 = np.multiply(2, KFC)
        cfn4 = np.multiply(2, np.multiply(mfactor, KFC))
        cfn5 = np.multiply(mfactor, np.square(KFC))
        correctionnumerator = np.add(np.subtract(np.subtract(np.subtract(2, cfn2), cfn3), cfn4), cfn5)
        correctiondenominator = np.multiply(KFC, (np.multiply(mfactor, (np.add(-2, KFC)))))
        correction = np.divide(correctionnumerator, correctiondenominator)
        # write secondary function
        # takes as input position, doesn't return anything and then execute with pool
        # with pool, nthreads as p
        # p.map(f, dns)
        
        def DiffCalc(dn):
            pos = dn-1
            dX = np.subtract(X[dn:], X[:len(X)-dn])
            dY = np.subtract(Y[dn:], Y[:len(Y)-dn])
            dZ = np.subtract(Z[dn:], Z[:len(Z)-dn])
            N[pos] = len(dX)
            denom = np.multiply(2, np.multiply(dn, TimeStep))
            DappX[pos] = np.divide(np.mean(np.square(dX)), denom)
            DappY[pos] = np.divide(np.mean(np.square(dY)), denom)
            DappZ[pos] = np.divide(np.mean(np.square(dZ)), denom)
            
            #correct for Kalman factor
            DcorrX[pos] = np.divide(DappX[pos], correction[pos])
            DcorrY[pos] = np.divide(DappY[pos], correction[pos])
            DcorrZ[pos] = np.divide(DappZ[pos], correction[pos])
            return
        
        pool = ThreadPool()
        pool.restart()
        
        pool.map(DiffCalc, dns)
        
        pool.close()
        pool.terminate()


        return DappX, DappY, DappZ, DcorrX, DcorrY, DcorrZ, N

    
    @staticmethod
    def MLE(tracks, start, end, volfraction):
        MLEres = {}

        def MLEModel(x, a, c): # define MLE model, model for Dcorr
                return np.add(a, np.divide(np.square(c), x))
            
        def MLEAppModel(x, a, c): # define MLE model, model for Dapp
                return np.subtract(a, np.multiply(a, np.exp(np.multiply(-c, x))))

        for key in tracks:
            print(f'MLE of track {key+1}/{len(tracks.keys())} started...')
            MLEres[key] = {}
            TimeStep = tracks[key]['params']['TimeStep']
            X = tracks[key]['XYZ'].values[:,0]
            Y = tracks[key]['XYZ'].values[:,1]
            Z = tracks[key]['XYZ'].values[:,2]
            KFC = 0.003; # Kalman filter coefficient
            m = 100./10; # Ratio of tracking time step to Kalman time step (100 us vs. 10 us)
            
            # Number of overlapping time lags
            #lags = end
            
            
                
            DappX, DappY, DappZ, DcorrX, DcorrY, DcorrZ, N = TrackAnalysis.MLEloop(X, Y, Z, end+1, m, KFC, TimeStep)
            
            # fitMax was found empirically to provide the most
            # consistent fitting results
            #delta = np.multiply(np.arange(1, len(DappX)+1).T, TimeStep)
        
            # fitMax was found empirically to provide the most
            # consistent fitting results
            fitMin = start
            fitMax = end
            DcorrX = DcorrX[fitMin:fitMax]
            DcorrY = DcorrY[fitMin:fitMax]
            DcorrZ = DcorrZ[fitMin:fitMax]
            N = N[fitMin:fitMax]
            
            MLEres[key]['DappX'] = DappX
            MLEres[key]['DappY'] = DappY
            MLEres[key]['DappZ'] = DappZ
            MLEres[key]['DcorrX'] = DcorrX
            MLEres[key]['DcorrY'] = DcorrY
            MLEres[key]['DcorrZ'] = DcorrZ
            MLEres[key]['N'] = N
    
            delta = np.multiply(np.arange(fitMin+1, fitMax+1).T, TimeStep)
            MLEres['delta'] = delta

            
    
            mmod = Model(MLEModel)
            fparx = mmod.make_params(a=np.mean(DcorrX[200:]), c=0.015)
            for fkey in fparx:
                fparx[fkey].set(min=0)
            fpary = mmod.make_params(a=np.mean(DcorrY[200:]), c=0.015)
            for fkey in fpary:
                fpary[fkey].set(min=0)
            fparz = mmod.make_params(a=np.mean(DcorrZ[200:]), c=0.015)
            for fkey in fparz:
                fparz[fkey].set(min=0)
    
            weights = N
            
            # Fits are performed for each axis, again according to the procedure given by the reference: Xu, C. S.; Cang, H.; Montiel, D.; Yang, H.. J. Phys. Chem. C 2007, 111 (1), 32–35.
            resultX = mmod.fit(DcorrX, params=fparx, x=delta, weights=weights)
            resultY = mmod.fit(DcorrY, params=fpary, x=delta, weights=weights)
            resultZ = mmod.fit(DcorrZ, params=fparz, x=delta, weights=weights)
            
            def R2(data, fit, residuals):
                meandiff = np.sum(np.square(np.subtract(data, np.mean(data))))
                residdiff = np.sum(np.square(residuals))
                r2 = np.abs(np.subtract(1, np.divide(residdiff, meandiff)))
                return r2
            
            R2X = R2(DcorrX, resultX.best_fit, np.divide(resultX.residual, N))
            R2Y = R2(DcorrY, resultY.best_fit, np.divide(resultY.residual, N))
            R2Z = R2(DcorrZ, resultZ.best_fit, np.divide(resultZ.residual, N))
            
            MLEres[key]['R2X'] = R2X
            MLEres[key]['R2Y'] = R2Y
            MLEres[key]['R2Z'] = R2Z
            
            print(f'X-R^2 of track {key+1} is {R2X}')
            print(f'Y-R^2 of track {key+1} is {R2Y}')
            print(f'Z-R^2 of track {key+1} is {R2Z}')
            
            DtrueX = resultX.best_values['a']
            SigmaX = resultX.best_values['c']
            SigmaXError = resultX.params['c'].stderr
            
            DtrueY = resultY.best_values['a']
            SigmaY = resultY.best_values['c']
            SigmaYError = resultY.params['c'].stderr
    
            DtrueZ = resultZ.best_values['a']
            SigmaZ = resultZ.best_values['c']
            SigmaZError = resultZ.params['c'].stderr
            
            MLEres[key]['DtrueX'] = DtrueX
            MLEres[key]['SigmaX'] = SigmaX
            MLEres[key]['SigmaXError'] = SigmaXError
            MLEres[key]['DtrueY'] = DtrueY
            MLEres[key]['SigmaY'] = SigmaY
            MLEres[key]['SigmaYError'] = SigmaYError
            MLEres[key]['DtrueZ'] = DtrueZ
            MLEres[key]['SigmaZ'] = SigmaZ
            MLEres[key]['SigmaZError'] = SigmaZError
            
            # Corrected Variance of diffusion coefficient MLE
            # Variance of diffusion coefficient MLE
            deltafactor = (np.multiply(N, np.multiply(np.multiply(KFC, m), np.square(delta))))
            
            squareSigX = np.square(SigmaX)
            squareSigY = np.square(SigmaY)
            squareSigZ = np.square(SigmaZ)
            
            var_DtrueX = np.divide(np.multiply(2, np.square(np.add(squareSigX, np.multiply(delta, DtrueX)))), deltafactor)
            var_DtrueY = np.divide(np.multiply(2, np.square(np.add(squareSigY, np.multiply(delta, DtrueY)))), deltafactor)
            var_DtrueZ = np.divide(np.multiply(2, np.square(np.add(squareSigZ, np.multiply(delta, DtrueZ)))), deltafactor)
            
            # find time lags which minimise variance in diffusion coefficient MLEs
            MinXIndex = np.nanargmin(var_DtrueX)
            MinYIndex = np.nanargmin(var_DtrueY)
            MinZIndex = np.nanargmin(var_DtrueZ)
            
            MLEres[key]['MinXIndex'] = MinXIndex
            MLEres[key]['MinYIndex'] = MinYIndex
            MLEres[key]['MinZIndex'] = MinZIndex
            
            MLEres[key]['var_DtrueX'] = var_DtrueX
            MLEres[key]['var_DtrueY'] = var_DtrueY
            MLEres[key]['var_DtrueZ'] = var_DtrueZ
    
            # Calculate translational diffusion parameters and their errors using the minimal variance time lags
            DXTrans = np.multiply(np.subtract(DcorrX[MinXIndex], np.divide(squareSigX, delta[MinXIndex])), 1e-12) # in m^2/s
            DXTranserror = np.multiply(np.sqrt(var_DtrueX[MinXIndex]), 1e-12) # in m^2/s
            DYTrans = np.multiply(np.subtract(DcorrY[MinYIndex], np.divide(squareSigY, delta[MinYIndex])), 1e-12) # in m^2/s
            DYTranserror = np.multiply(np.sqrt(var_DtrueY[MinYIndex]), 1e-12) # in m^2/s
            DZTrans = np.multiply(np.subtract(DcorrX[MinZIndex], np.divide(squareSigZ, delta[MinZIndex])), 1e-12) # in m^2/s
            DZTranserror = np.multiply(np.sqrt(var_DtrueZ[MinZIndex]), 1e-12) # in m^2/s
            DXYZTrans = np.mean([DXTrans, DYTrans, DZTrans]) # in m^2/s
            DXYZTranserror = np.divide(np.linalg.norm([DXTranserror, DYTranserror, DZTranserror]), len([DXTranserror, DYTranserror, DZTranserror]))
            
            MLEres[key]['DXTrans'] = DXTrans
            MLEres[key]['DXTranserror'] = DXTranserror
            MLEres[key]['DYTrans'] = DYTrans
            MLEres[key]['DYTranserror'] = DYTranserror
            MLEres[key]['DZTrans'] = DZTrans
            MLEres[key]['DZTranserror'] = DZTranserror
            MLEres[key]['DXYZTrans'] = DXYZTrans
            MLEres[key]['DXYZTranserror'] = DXYZTranserror
            
            # do focal shift correction
            from OpticsFunctions import OpticsFunc
            OF = OpticsFunc()
            
            DcorrZ_FSC = np.multiply(np.square(OF.MeanFocalShift(np.arange(400, 701),0.7,1.485, volfraction)), DcorrZ)
            fparz = mmod.make_params(a=np.mean(DcorrZ_FSC[200:]), c=0.015)
            for fkey in fparz:
                fparz[fkey].set(min=0)
            
            resultFSCZ = mmod.fit(DcorrZ_FSC, params=fparz, x=delta, weights=weights)
            
            MLEres[key]['chisquareZFSC'] = R2(DcorrZ_FSC, resultFSCZ.best_fit, np.divide(resultFSCZ.residual, N))
    
            DtrueZFSC = resultFSCZ.best_values['a']
            SigmaZFSC = resultFSCZ.best_values['c']
            SigmaZFSCError = resultFSCZ.params['c'].stderr
    
            squareSigZFSC = np.square(SigmaZFSC)
    
            MLEres[key]['DtrueZFSC'] = DtrueZFSC
            MLEres[key]['SigmaZFSC'] = SigmaZFSC
            MLEres[key]['SigmaZFSCError'] = SigmaZFSCError
    
            var_DtrueZFSC = np.divide(np.multiply(2, np.square(np.add(squareSigZFSC, np.multiply(delta, DtrueZFSC)))), deltafactor)        
            
            MinZFSCIndex = np.nanargmin(var_DtrueZFSC)
            
            MLEres[key]['MinZFSCIndex'] = MinZFSCIndex
            MLEres[key]['var_DtrueZFSC'] = var_DtrueZFSC
            
            DZFSCTrans = np.multiply(np.subtract(DcorrX[MinZFSCIndex], np.divide(squareSigZFSC, delta[MinZFSCIndex])), 1e-12) # in m^2/s
            DZFSCTranserror = np.multiply(np.sqrt(var_DtrueZ[MinZFSCIndex]), 1e-12) # in m^2/s
            DXYZFSCTrans = np.mean([DXTrans, DYTrans, DZFSCTrans]) # in m^2/s
            DXYZFSCTranserror = np.divide(np.linalg.norm([DXTranserror, DYTranserror, DZFSCTranserror]), len([DXTranserror, DYTranserror, DZFSCTranserror]))
            
            MLEres[key]['DZFSCTrans'] = DZFSCTrans
            MLEres[key]['DZFSCTranserror'] = DZFSCTranserror
            MLEres[key]['DXYZFSCTrans'] = DXYZFSCTrans
            MLEres[key]['DXYZFSCTranserror'] = DXYZFSCTranserror
            
        return MLEres
    
    @staticmethod
    def PIDAnalysis(tracks):
        PIDRes = {}
        trackset = deepcopy(tracks)
        for key in trackset:
            print(f'MLE of track {key+1} started...')
            PIDRes[key] = {}
            TimeStep = trackset[key]['params']['TimeStep']
            PIDRes[key]['PID'] = trackset[key]['params']['PID']
            X = trackset[key]['XYZ'].values[:,0]
            Y = trackset[key]['XYZ'].values[:,1]
            Z = trackset[key]['XYZ'].values[:,2]
            KFC = 0.003; # Kalman filter coefficient
            m = 100./10; # Ratio of tracking time step to Kalman time step (20 us vs. 10 us)
            
            # Number of overlapping time lags
            lags = len(X)-1
            
            
                
            DappX, DappY, DappZ, DcorrX, DcorrY, DcorrZ, N = TrackAnalysis.MLEloop(X, Y, Z, lags, m, KFC, TimeStep)
            
            # fitMax was found empirically to provide the most
            # consistent fitting results
            delta = np.multiply(np.arange(1, len(DappX)+1).T, TimeStep)
            
            PIDRes[key]['DappX'] = DappX
            PIDRes[key]['DappY'] = DappY
            PIDRes[key]['DappZ'] = DappZ
            PIDRes[key]['DcorrX'] = DcorrX
            PIDRes[key]['DcorrY'] = DcorrY
            PIDRes[key]['DcorrZ'] = DcorrZ
            PIDRes[key]['N'] = N
    
            
            PIDRes[key]['delta'] = delta
        print("MLEs finished.")
        return PIDRes
    