'''
Created on Aug 12, 2016

@author: urishaham
'''
#from KerasExtensions import import 
import os
from keras import backend as K
import numpy as np
import CostFunctions as cf

def pause():
    programPause = input("Press the <ENTER> key to continue...")
    
def checkScale(targetSample, outputSample, scale,  nIters = 3, batchSize = 1000):
    mmd_TT = np.zeros(nIters) 
    mmd_OT = np.zeros(nIters)
    #ratios = np.zeros(nIters)     
    for i in range(nIters):    
        T = targetSample[np.random.randint(targetSample.shape[0], size=batchSize),:]
        T1 = targetSample[np.random.randint(targetSample.shape[0], size=batchSize),:]
        T2 = targetSample[np.random.randint(targetSample.shape[0], size=batchSize),:]
        O = outputSample[np.random.randint(outputSample.shape[0], size=batchSize),:]
        mmd_TT[i] = K.eval(cf.MMD(T1,T2, scales=[scale]).cost(K.variable(value=T1), K.variable(value=T2)))
        mmd_OT[i] = K.eval(cf.MMD(T,O, scales=[scale]).cost(K.variable(value=T), K.variable(value=O)))
        #ratios[i] = (mmd_OT[i] - mmd_TT[i])/ mmd_OT[i]
    print('scale: ' + str(scale))
    print('mmd_TT: ' + str (np.mean(mmd_TT)))
    print('mmd_OT: ' + str (np.mean(mmd_OT)))
    ratio = (np.mean(mmd_OT) - np.mean(mmd_TT))/ np.mean(mmd_OT)
    print('ratio: ' + str(ratio))
    return np.mean(mmd_TT), np.mean(mmd_OT), ratio
    
def checkScales(targetSample, outputSample, scales, nIters = 3):
    TT = np.zeros(len(scales))
    OT = np.zeros(len(scales))
    ratios = np.zeros(len(scales))
    for i in range (len(scales)):
        TT[i], OT[i], ratios[i] =  checkScale(targetSample, outputSample, scales[i], nIters)
    chosenScale = scales[np.argmax(ratios)]    
    print('most distinguishing scale: '+str(chosenScale))
    return TT, OT, ratios

def permute(X1, X2, numPts = 1000):
    n1,d = X1.shape
    n2 = X2.shape[0]
    Y1 = np.zeros((numPts,d))
    Y2 = np.zeros((numPts,d))
    for i in range(numPts):
        set = np.random.randint(2)
        if set==0:
            ind = np.random.randint(n1)
            pt = X1[ind]
        else:
            ind = np.random.randint(n2)
            pt = X2[ind]
        Y1[i] = pt
        set = np.random.randint(2)
        if set==0:
            ind = np.random.randint(n1)
            pt = X1[ind]
        else:
            ind = np.random.randint(n2)
            pt = X2[ind]
        Y2[i] = pt   
        return Y1, Y2      