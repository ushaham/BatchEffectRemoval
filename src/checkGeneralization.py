'''
Created on Sep 13, 2016

@author: Uri Shaham
'''

import os.path
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from keras.models import Model
from keras import callbacks as cb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
from keras import backend as K
from numpy import genfromtxt
import sklearn.preprocessing as prep
from keras import initializations
from keras.layers import Input, Dense, merge, Activation
from keras.layers.normalization import BatchNormalization

denoise = True # wether or not to train a denoising autoencoder to remover the zeros
keepProb=.8

# AE confiduration
ae_encodingDim = 25
l2_penalty_ae = 1e-3 

#MMD net configuration
mmdNetLayerSizes = [25, 25]
l2_penalty = 5e-3
init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 25.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)



######################
###### get data ######
######################
# we load two CyTOF samples 
# Labels are cell types

sample1APath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1.csv')
sample1BPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2.csv')
sample2APath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1.csv')
sample2BPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2.csv')


source1 = genfromtxt(sample1APath, delimiter=',', skip_header=0)
target1 = genfromtxt(sample1BPath, delimiter=',', skip_header=0)
source2 = genfromtxt(sample2APath, delimiter=',', skip_header=0)
target2 = genfromtxt(sample2BPath, delimiter=',', skip_header=0)

# pre-process data: log transformation, a standard practice with CyTOF data
target1 = dh.preProcessCytofData(target1)
source1 = dh.preProcessCytofData(source1) 
target2 = dh.preProcessCytofData(target1)
source2 = dh.preProcessCytofData(source1) 

numZerosOK=1
toKeepS1 = np.sum((source1==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS1))
toKeepT1 = np.sum((target1==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT1))
toKeepS2 = np.sum((source2==0), axis = 1) <=numZerosOK
print(np.sum(toKeepS2))
toKeepT2 = np.sum((target2==0), axis = 1) <=numZerosOK
print(np.sum(toKeepT2))

inputDim = target1.shape[1]

if denoise:
    trainTarget_ae = np.concatenate([source1[toKeepS1], target1[toKeepT1], source2[toKeepS2], target2[toKeepT2]], axis=0)
    trainData_ae = trainTarget_ae * np.random.binomial(n=1, p=keepProb, size = trainTarget_ae.shape)
    input_cell = Input(shape=(inputDim,))
    encoded = Dense(ae_encodingDim, activation='relu',W_regularizer=l2(l2_penalty_ae))(input_cell)
    decoded = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty_ae))(encoded)
    autoencoder = Model(input=input_cell, output=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(trainData_ae, trainTarget_ae, nb_epoch=500, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=10,  mode='auto')])    
    source1 = autoencoder.predict(source1)
    target1 = autoencoder.predict(target1)
    source2 = autoencoder.predict(source1)
    target2 = autoencoder.predict(target1)

# rescale the data to have zero mean and unit variance
preprocessor = prep.StandardScaler().fit(source1)
source1 = preprocessor.transform(source1)  
target1 = preprocessor.transform(target1)    

# rescale the data to have zero mean and unit variance
preprocessor = prep.StandardScaler().fit(source2)
source2 = preprocessor.transform(source2)  
target2 = preprocessor.transform(target2)    


#############################################
######## train net on each new sample  ######
#############################################
# the nets use the configuration defined above

# first net
calibInput_1 = Input(shape=(inputDim,))
block1_bn1_1 = BatchNormalization()(calibInput_1)
block1_a1_1 = Activation('relu')(block1_bn1_1)
block1_w1_1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a1_1) 
block1_bn2_1 = BatchNormalization()(block1_w1_1)
block1_a2_1 = Activation('relu')(block1_bn2_1)
block1_w2_1 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a2_1) 
block1_output_1 = merge([block1_w2_1, calibInput_1], mode = 'sum')
block2_bn1_1 = BatchNormalization()(block1_output_1)
block2_a1_1 = Activation('relu')(block2_bn1_1)
block2_w1_1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a1_1) 
block2_bn2_1 = BatchNormalization()(block2_w1_1)
block2_a2_1 = Activation('relu')(block2_bn2_1)
block2_w2_1 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a2_1) 
block2_output_1 = merge([block2_w2_1, block1_output_1], mode = 'sum')

calibMMDNet_1 = Model(input=calibInput_1, output=block2_output_1)
optimizer = keras.optimizers.rmsprop(lr=0.0)
calibMMDNet_1.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_output_1,target1,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source1.shape[0])
calibMMDNet_1.fit(source1, sourceLabels[:source1.shape[0]],nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(source1, target1, calibMMDNet_1.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=20,mode='auto')])


# second net
calibInput_2 = Input(shape=(inputDim,))
block1_bn1_2 = BatchNormalization()(calibInput_2)
block1_a1_2 = Activation('relu')(block1_bn1_2)
block1_w1_2 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a1_2) 
block1_bn2_2 = BatchNormalization()(block1_w1_2)
block1_a2_2 = Activation('relu')(block1_bn2_2)
block1_w2_2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block1_a2_2) 
block1_output_2 = merge([block1_w2_1, calibInput_2], mode = 'sum')
block2_bn1_2 = BatchNormalization()(block1_output_2)
block2_a1_2 = Activation('relu')(block2_bn1_1)
block2_w1_2 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a1_2) 
block2_bn2_2 = BatchNormalization()(block2_w1_2)
block2_a2_2 = Activation('relu')(block2_bn2_2)
block2_w2_2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = init)(block2_a2_2) 
block2_output_2 = merge([block2_w2_2, block1_output_2], mode = 'sum')

calibMMDNet_2 = Model(input=calibInput_2, output=block2_output_2)
optimizer = keras.optimizers.rmsprop(lr=0.0)
calibMMDNet_2.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_output_2,target2,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source2.shape[0])
calibMMDNet_1.fit(source2, sourceLabels[:source2.shape[0]],nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(source2, target2, calibMMDNet_2.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=20,mode='auto')])



###############################################
######## evaluate generalization ability ######
###############################################

calibration_11 =  calibMMDNet_1.predict(source1)
calibration_22 =  calibMMDNet_2.predict(source2)
calibration_12 =  calibMMDNet_1.predict(source2)
calibration_21 =  calibMMDNet_2.predict(source1)

# our hope is that:
#     calibration_b1 is as similar to newTarget1 as calibration_a1
#     calibration_a2 is as similar to newTarget2 as calibration_b2

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA(n_components=2)
pca.fit(target1)
target_sample_pca = np.dot(target1, pca.components_[[0,1]].transpose())
projection_org = np.dot(calibration_11, pca.components_[[0,1]].transpose())
projection_cross = np.dot(calibration_21, pca.components_[[0,1]].transpose())

# plot of 
fig = plt.figure()
a1 = fig.add_subplot(211)
a1.scatter(target_sample_pca[:,0], target_sample_pca[:,1], color = 'blue', s=1)
a1.scatter(projection_org[:,0], projection_org[:,1], color='red', s=1)
a1.set_title("patient 1,  target (blue), net 1 calib  (red)")

a2 = fig.add_subplot(212)
a2.scatter(target_sample_pca[:,0], target_sample_pca[:,1], color = 'blue', s=1)
a2.scatter(projection_cross[:,0], projection_cross[:,1], color='green', s=1)
a2.set_title("patient 1,  target (blue), net 2 calib (green) ")

plt.draw()

pca = decomposition.PCA(n_components=2)
pca.fit(target2)
target_sample_pca = np.dot(target2, pca.components_[[0,1]].transpose())
projection_org = np.dot(calibration_22, pca.components_[[0,1]].transpose())
projection_cross = np.dot(calibration_12, pca.components_[[0,1]].transpose())

# plot of 
fig = plt.figure()
a1 = fig.add_subplot(211)
a1.scatter(target_sample_pca[:,0], target_sample_pca[:,1], color = 'blue', s=1)
a1.scatter(projection_org[:,0], projection_org[:,1], color='red', s=1)
a1.set_title("patient 2,  target (blue), net 2 calib  (red)")

a2 = fig.add_subplot(212)
a2.scatter(target_sample_pca[:,0], target_sample_pca[:,1], color = 'blue', s=1)
a2.scatter(projection_cross[:,0], projection_cross[:,1], color='green', s=1)
a2.set_title("patient 2,  target (blue), net 1 calib (green) ")

plt.draw()


##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = source1.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target1.shape[0], size = 1000)


mmd_before = K.eval(cf.MMD(source1,target1).cost(K.variable(value=source1[sourceInds]), K.variable(value=target1[targetInds])))
mmd_after_a1 = K.eval(cf.MMD(source1,target1).cost(K.variable(value=calibration_11[sourceInds]), K.variable(value=target1[targetInds])))
mmd_after_b1 = K.eval(cf.MMD(source1,target1).cost(K.variable(value=calibration_21[sourceInds]), K.variable(value=target1[targetInds])))

print('patient 1: MMD before calibration: ' + str(mmd_before))
print('patient 1: MMD after calibration (net a): ' + str(mmd_after_a1))
print('patient 1: MMD after calibration (net b): ' + str(mmd_after_b1))

sourceInds = np.random.randint(low=0, high = source2.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target2.shape[0], size = 1000)


mmd_before = K.eval(cf.MMD(source2,target2).cost(K.variable(value=source2[sourceInds]), K.variable(value=target2[targetInds])))
mmd_after_a2 = K.eval(cf.MMD(source2,target2).cost(K.variable(value=calibration_12[sourceInds]), K.variable(value=target2[targetInds])))
mmd_after_b2 = K.eval(cf.MMD(source2,target2).cost(K.variable(value=calibration_22[sourceInds]), K.variable(value=target2[targetInds])))

print('patient 2: MMD before calibration: ' + str(mmd_before))
print('patient 2: MMD after calibration (net b): ' + str(mmd_after_b2))
print('patient 2: MMD after calibration (net a): ' + str(mmd_after_a2))

'''
patient 1: MMD before calibration: 0.377964
patient 1: MMD after calibration (net a): 0.14309
patient 1: MMD after calibration (net b): 0.147311

patient 2: MMD before calibration: 0.431609
patient 2: MMD after calibration (net b): 0.15522
patient 2: MMD after calibration (net a): 0.156867


'''