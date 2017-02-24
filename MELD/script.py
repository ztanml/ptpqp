# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:24:06 2016

@author: zhaoshiwen
"""

import numpy as np

import MELD as MELD
reload(MELD)


n = 1000
Y = np.loadtxt('../data/CTp20Yn' + str(n) + '.txt')
(p,n) = Y.shape

Yt = np.array([0]*p) 
# the type of the variables
# 0: categorical
# 1: non-categorical with non-restrictive mean
# 2: non-categorical with positive mean


k = 3 # the number of components
S = 100 # maximum number of iterations


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

# --------------------------------
# --------------------------------
# Parameter estimation using second moment matrices

# create an object of MELD class
myMELD = MELD.MELD(Y,Yt,k)

# calculate third moment tensors
myMELD.calM3()
myMELD.calM3_bar()

# ------------ first stage
# initialize weight matrices to identity
myMELD.initializeWeight_M3()

# start to perform first stage estimation
Result = myMELD.estimatePhiGrad_M2M3(S)

# ------------ second stage
# recalculate weight matrix
myMELD.updateWeight_M2()
myMELD.updateWeight_M3()

# start to performo second stage estimation
Results = myMELD.estimatePhiGrad_M2M3(S,step = 0.1)
