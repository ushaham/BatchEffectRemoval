'''
Created on Dec 5, 2016

@author: urishaham
'''

import os.path
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from Calibration_Util import Misc
from keras.layers import Input, Dense, merge, Activation
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
from keras.layers.normalization import BatchNormalization
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
from keras import backend as K
import ScatterHist as sh
from statsmodels.distributions.empirical_distribution import ECDF
from keras import initializations
from numpy import genfromtxt
import sklearn.preprocessing as prep

# configuration hyper parameters
denoise = True # wether or not to train a denoising autoencoder to remover the zeros
keepProb=.8

# AE confiduration
ae_encodingDim = 25
l2_penalty_ae = 1e-3 

#MMD net configuration
mmdNetLayerSizes = [25, 25]
l2_penalty = 5e-3
init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)


######################
###### get data ######
######################
# we load two CyTOF samples 

sampleAPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1.csv')
sampleBPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2.csv')

source = genfromtxt(sampleAPath, delimiter=',', skip_header=0)
target = genfromtxt(sampleBPath, delimiter=',', skip_header=0)

# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source) 

numZerosOK=1
toKeepS = np.sum((source==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS))
toKeepT = np.sum((target==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT))

inputDim = target.shape[1]

if denoise:
    trainTarget_ae = np.concatenate([source[toKeepS], target[toKeepT]], axis=0)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, nb_epoch=500, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=25,  mode='auto')])    
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
target = preprocessor.transform(target)    

#############################
######## train MMD net ######
#############################


calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a2) 
block1_output = merge([block1_w2, calibInput], mode = 'sum')
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a2) 
block2_output = merge([block2_w2, block1_output], mode = 'sum')

calibMMDNet = Model(input=calibInput, output=block2_output)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

#train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source.shape[0])
calibMMDNet.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

##############################
###### evaluate results ######
##############################

calibratedSource = calibMMDNet.predict(source)

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)

##################################### qualitative evaluation: per-marker empirical cdfs #####################################

for i in range(target.shape[1]):
    targetMarker = target[:,i]
    beforeMarker = source[:,i]
    afterMarker = calibratedSource[:,i]
    m = np.min([np.min(targetMarker), np.min(beforeMarker), np.min(afterMarker)])
    M = np.max([np.max(targetMarker), np.max(beforeMarker), np.max(afterMarker)])
    x = np.linspace(m, M, num=100)
    target_ecdf = ECDF(targetMarker)
    before_ecdf = ECDF(beforeMarker)
    after_ecdf = ECDF(afterMarker)   
    tgt_ecdf = target_ecdf(x)
    bf_ecdf = before_ecdf(x)
    af_ecdf = after_ecdf(x)    
    fig = plt.figure()
    a1 = fig.add_subplot(111)
    a1.plot(tgt_ecdf, color = 'blue') 
    a1.plot(bf_ecdf, color = 'red') 
    a1.plot(af_ecdf, color = 'green') 
    a1.set_xticklabels([])
    plt.legend(['target', 'before calibration', 'after calibration'], loc=0)
    plt.show()
       
##################################### Correlation matrices ##############################################
corrB = np.corrcoef(source, rowvar=0)
corrA = np.corrcoef(calibratedSource, rowvar=0)
corrT = np.corrcoef(target, rowvar=0)
FB = corrT - corrB
FA = corrT - corrA
NB = np.linalg.norm(FB, 'fro')
NA = np.linalg.norm(FA, 'fro')

print('norm before calibration: ', str(NB))
print('norm after calibration: ', str(NA)) 

fa = FA.flatten()
fb = FB.flatten()

f = np.zeros((fa.shape[0],2))
f[:,0] = fb
f[:,1] = fa

fig = plt.figure()
plt.hist(f, bins = 20, normed=True, histtype='bar')
plt.legend(['before calib.', 'after calib.'], loc=1)
plt.yticks([])
plt.show()
##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(block2_output,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
mmd_after = K.eval(cf.MMD(block2_output,target).cost(K.variable(value=calibratedSource[sourceInds]), K.variable(value=target[targetInds])))
print('MMD before calibration: ' + str(mmd_before))
print('MMD after calibration: ' + str(mmd_after))


##################################### quantitative evaluation: MMD #####################################
# MMD with the a single scale at a time, for various scales
scales = [3e-1, 1e-0, 3e-0, 1e1]
TT_b, OT_b, ratios_b = Misc.checkScales(target, source, scales)
TT_a, OT_a, ratios_a = Misc.checkScales(target, calibratedSource, scales)
print('scales: ', scales)
print('MMD(target,target): ', TT_a)
print('MMD(before calibration, target): ', OT_b)
print('MMD(after calibration, target): ', OT_a)

'''
 this script gave: 
norm before calibration:  3.59862372511
norm after calibration:  1.98529712509

MMD before calibration: 0.721767
MMD after calibration: 0.162444

MMD(target,target):              [ 0.04440335  0.0442343   0.03732886  0.02267724]
MMD(before calibration, target); [ 0.05664794  0.07511662  0.33642591  0.43417276]
MMD(after calibration, target):  [ 0.05683116  0.05906311  0.0479396   0.0240008 ]
'''
