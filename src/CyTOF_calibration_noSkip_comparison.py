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
from keras.callbacks import History 

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
init_ns = 'glorot_normal'


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

####################################
######## train MMD net noSkip ######
####################################


calibInput_ns = Input(shape=(inputDim,))
block1_bn1_ns = BatchNormalization()(calibInput_ns)
block1_a1_ns = Activation('relu')(block1_bn1_ns)
block1_w1_ns = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = init_ns)(block1_a1_ns) 
block1_bn2_ns = BatchNormalization()(block1_w1_ns)
block1_a2_ns = Activation('relu')(block1_bn2_ns)
block1_w2_ns = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init_ns)(block1_a2_ns) 
#block1_output = merge([block1_w2, calibInput], mode = 'sum')
block2_bn1_ns = BatchNormalization()(block1_w2_ns)
block2_a1_ns = Activation('relu')(block2_bn1_ns)
block2_w1_ns = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = init_ns)(block2_a1_ns) 
block2_bn2_ns = BatchNormalization()(block2_w1_ns)
block2_a2_ns = Activation('relu')(block2_bn2_ns)
block2_w2_ns = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init_ns)(block2_a2_ns) 
#block2_output_ns = merge([block2_w2_ns, block1_output_ns], mode = 'sum')

calibMMDNet_noSkip = Model(input=calibInput_ns, output=block2_w2_ns)

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

calibMMDNet_noSkip.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_w2_ns,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source.shape[0])
history_noSkip = History()
calibMMDNet_noSkip.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(source, target, calibMMDNet_noSkip.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto'), history_noSkip])

###################################
######## train MMD net       ######
###################################


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


#train MMD net
lrate = LearningRateScheduler(step_decay)
optimizer = keras.optimizers.rmsprop(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source.shape[0])
history = History()
calibMMDNet.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto'), history])




##############################
###### evaluate results ######
##############################
calibratedSource_noSkip = calibMMDNet_noSkip.predict(source)
calibratedSource = calibMMDNet.predict(source)

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after = pca.transform(calibratedSource)
projection_after_noSkip = pca.transform(calibratedSource_noSkip)

# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_noSkip[:,pc1], projection_after_noSkip[:,pc2], axis1, axis2)

       
##################################### Correlation matrices ##############################################
corrB = np.corrcoef(source, rowvar=0)
corrA = np.corrcoef(calibratedSource, rowvar=0)
corrA_noSkip = np.corrcoef(calibratedSource_noSkip, rowvar=0)
corrT = np.corrcoef(target, rowvar=0)
FB = corrT - corrB
FA = corrT - corrA
FA_noSkip = corrT - corrA_noSkip
NB = np.linalg.norm(FB, 'fro')
NA = np.linalg.norm(FA, 'fro')
NA_noSkip = np.linalg.norm(FA_noSkip, 'fro')


print('norm before calibration: ', str(NB))
print('norm after calibration: ', str(NA)) 
print('norm after calibration (no skip connections): ', str(NA_noSkip)) 

fa = FA.flatten()
fa_noSkip = FA_noSkip.flatten()
fb = FB.flatten()

f = np.zeros((fa.shape[0],3))
f[:,0] = fb
f[:,1] = fa
f[:,2] = fa_noSkip

fig = plt.figure()
plt.hist(f, bins = 10, normed=True, histtype='bar')
plt.legend(['before calibration', 'ResNet', 'standard MLP'], loc=1)
plt.show()


##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(block2_w2,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
mmd_after = K.eval(cf.MMD(block2_output,target).cost(K.variable(value=calibratedSource[sourceInds]), K.variable(value=target[targetInds])))
mmd_after_noSkip = K.eval(cf.MMD(block2_w2,target).cost(K.variable(value=calibratedSource_noSkip[sourceInds]), K.variable(value=target[targetInds])))
print('MMD before calibration: ' + str(mmd_before))
print('MMD after calibration: ' + str(mmd_after))
print('MMD after calibration (no skip connections): ' + str(mmd_after_noSkip))


##################################### quantitative evaluation: MMD #####################################
# MMD with the a single scale at a time, for various scales
scales = [3e-1, 1e-0, 3e-0, 1e1]
TT_b, OT_b, ratios_b = Misc.checkScales(target, source, scales)
TT_a, OT_a, ratios_a = Misc.checkScales(target, calibratedSource, scales)
TT_a_noSkip, OT_a_noSkip, ratios_a_noSkip = Misc.checkScales(target, calibratedSource_noSkip, scales)
print('scales: ', scales)
print('MMD(target,target): ', TT_a)
print('MMD(before calibration, target): ', OT_b)
print('MMD(after calibration, target): ', OT_a)
print('MMD(after calibration (no skip connections), target): ', OT_a_noSkip)

'''
 this script gave: 
norm before calibration:                       3.19909245864
norm after calibration:                        1.45680005805 
norm after calibration (no skip connections):  2.29659227473

MMD before calibration: 0.756756
MMD after calibration:  0.189666
MMD after calibration (no skip connections): 0.171696

MMD(target,target):                                    [ 0.04500784  0.04569531  0.04027065  0.01983629]
MMD(before calibration, target):                       [ 0.05675488  0.06760406  0.32243539  0.45344398]
MMD(after calibration, target):                        [ 0.05660406  0.06103335  0.06739668  0.03540214]
MMD(after calibration (no skip connections), target):  [ 0.05697683  0.06160334  0.06223213  0.04051251]
'''

################### compare validation losses with and without skip connections ###################
resNet_valLoss = np.asarray(history.history['val_loss'])
noSkip_valLoss = np.asarray(history_noSkip.history['val_loss'])

x_ns = range(1,noSkip_valLoss.shape[0]+1)
x = range(1,resNet_valLoss.shape[0]+1)
fig = plt.figure()
plt.plot(x, resNet_valLoss, 'b--', x_ns, noSkip_valLoss, 'g--')
plt.xlabel('epoch', size=15)
plt.ylabel('validation loss', size=15)
plt.legend(['ResNet', 'standard MLP'], loc=1)
plt.show()

