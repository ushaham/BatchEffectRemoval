'''
Created on Dec 5, 2016

@author: urishaham
'''

import os.path
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import CostFunctions as cf
from keras import backend as K

from numpy import genfromtxt


orgDataPath = '/raid3/DropSeq/Retina/Second/1215/Data2_standardized_37PCs.csv'   
MMD_calibDataPath = '/raid3/uri/RNA_second_calibratedData1215.csv'    
CombatCalibDatapath = '/raid3/DropSeq/Retina/Second/Data2_CombatStandardized_37PCs.csv'
ZCalibDatapath = '/raid3/DropSeq/Retina/Second/Data2_ZscoreStandardized_37PCs.csv'   
pcaCalibDatapath = '/raid3/DropSeq/Retina/Second/Data2_RmPCsStandardized_34PCs.csv'   

orgData = genfromtxt(orgDataPath, delimiter=',', skip_header=0)
MMD_calibData = genfromtxt(MMD_calibDataPath, delimiter=',', skip_header=0)
Combat_calibData = genfromtxt(CombatCalibDatapath, delimiter=',', skip_header=0)
Z_calibData = genfromtxt(ZCalibDatapath, delimiter=',', skip_header=0)
PCA_calibData = genfromtxt(pcaCalibDatapath, delimiter=',', skip_header=0)
  
batchesPath = '/raid3/DropSeq/Retina/Second/1215/batch.csv'  
batches = genfromtxt(batchesPath, delimiter=',', skip_header=0) 

n_source = np.sum(batches == 2)
n_target = np.sum(batches == 1)

source = orgData[batches == 2]
target = orgData[batches == 1]
resNetCalibSource = MMD_calibData[batches == 2]
CombatCalibSource = Combat_calibData[batches == 2]
CombatCalibTarget = Combat_calibData[batches == 1]
ZCalibSource = Z_calibData[batches == 2]
PCA_CalibSource = PCA_calibData[batches == 2]
PCA_CalibTarget = PCA_calibData[batches == 1]


mmd_before = np.zeros(5)
mmd_after_resNet = np.zeros(5)
mmd_after_Combat = np.zeros(5)
mmd_after_Z = np.zeros(5)
mmd_after_PCA = np.zeros(5)
mmd_target_target = np.zeros(5)

for i in range(5):
    targetInds = np.random.randint(low=0, high = n_target, size = 1000)
    sourceInds = np.random.randint(low=0, high = n_source, size = 1000)
    targetInds1 = np.random.randint(low=0, high = n_target, size = 1000)
    mmd_before[i] = K.eval(cf.MMD(source,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_resNet[i] = K.eval(cf.MMD(resNetCalibSource,target).cost(K.variable(value=resNetCalibSource[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_Combat[i] = K.eval(cf.MMD(CombatCalibSource,CombatCalibTarget).cost(K.variable(value=CombatCalibSource[sourceInds]), K.variable(value=CombatCalibTarget[targetInds])))
    mmd_after_Z[i] = K.eval(cf.MMD(ZCalibSource,target).cost(K.variable(value=ZCalibSource[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_PCA[i] = K.eval(cf.MMD(PCA_CalibSource,PCA_CalibTarget).cost(K.variable(value=PCA_CalibSource[sourceInds]), K.variable(value=PCA_CalibTarget[targetInds])))
    mmd_target_target[i] = K.eval(cf.MMD(target,target).cost(K.variable(value=target[targetInds]), K.variable(value=target[targetInds1])))


print('MMD before calibration:         ' + str(np.mean(mmd_before))+'pm '+str(np.std(mmd_before)))
print('MMD after calibration (resNet): ' + str(np.mean(mmd_after_resNet))+'pm '+str(np.std(mmd_after_resNet)))
print('MMD after calibration (Combat): ' + str(np.mean(mmd_after_Combat))+'pm '+str(np.std(mmd_after_Combat)))
print('MMD after calibration (Z):      ' + str(np.mean(mmd_after_Z))+'pm '+str(np.std(mmd_after_Z)))
print('MMD after calibration (PCA):    ' + str(np.mean(mmd_after_PCA))+'pm '+str(np.std(mmd_after_PCA)))
print('MMD target-target:              ' + str(np.mean(mmd_target_target))+'pm '+str(np.std(mmd_target_target)))

'''
MMD before calibration:         0.436700809002pm 0.0047024403119
MMD after calibration (resNet): 0.127874596417pm 0.00625510858653
MMD after calibration (Combat): 0.158756631613pm 0.00749086873726
MMD after calibration (Z):      0.256576797366pm 0.00532713034745
MMD after calibration (PCA):    0.211824476719pm 0.010718020682
MMD target-target:              0.114321789145pm 0.00166756124391

'''

