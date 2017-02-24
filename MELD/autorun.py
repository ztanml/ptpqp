# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:24:06 2016

@author: zhaoshiwen
"""

import numpy as np
import MELD as MELD

N = [100,200,500,1000,5000];
K = [3,5,10,20];
C = [0,0.05,0.1];
S = 1000 # maximum number of iterations

for x in N:
    for k in K:
        for c in C:
            reload(MELD)
            fn = '../data/k' + str(k) + 'n' + str(x) + 'c' + str(c) + '.txt'
            print(fn)
            Y = np.loadtxt(fn)
            (p,n) = Y.shape
            Yt = np.array([0]*p) 

            # the type of the variables
            # 0: categorical
            # 1: non-categorical with non-restrictive mean
            # 2: non-categorical with positive mean

            # --------------------------------
            # --------------------------------
            # Parameter estimation using second moment matrices

            # create an object of MELD class
            myMELD = MELD.MELD(Y,Yt,k)
            # calculate second moment matrices
            myMELD.calM2()
            myMELD.calM2_bar()

            # ------------- first stage
            # initialize weight matrices to identity
            myMELD.initializeWeight_M2()

            # start to perform first stage estimation
            Result = myMELD.estimatePhiGrad_M2(S)

            # ------------- second stage
            # recalculate weight matrix
            myMELD.updateWeight_M2()

            # start to perform second stage estimation
            Results = myMELD.estimatePhiGrad_M2(S,step = 0.1)

            niter = Results['iter']
            last = Results['PHI'][niter-1]

            f = open('../data/MELDd4p25k' + str(k) + 'n' + str(x) + 'c' + str(c) + '.txt', 'w+')

            for j in range(25):
                for t in range(4):
                    for s in range(k):
                        f.write(str(last[j][s][t]) + ' ')
                    f.write('\n')
            f.close()
