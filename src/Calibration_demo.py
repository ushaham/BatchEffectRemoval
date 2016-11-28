'''
Created on Sep 13, 2016

@author: Uri Shaham
'''

import os.path
import keras.optimizers
from Calibration_Util import DataHandler as dh 
from Calibration_Util import FileIO as io
from keras.layers import Input, Dense, merge, BatchNormalization, Activation
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
import ScatterHist as sh
from statsmodels.distributions.empirical_distribution import ECDF
from keras import initializations

# space
space = 'data' # whether or not we want to train an autoencoder

######################
###### get data ######
######################
# we load two CyTOF samples 
# Labels are cell types, they are not used in the calibration.

sample1Path = os.path.join(io.DeepLearningRoot(),'Data/sampleA.csv')
sample1LabelsPath = os.path.join(io.DeepLearningRoot(),'Data/labelsA.csv')
sample2Path = os.path.join(io.DeepLearningRoot(),'Data/sampleB.csv')
sample2LabelsPath = os.path.join(io.DeepLearningRoot(),'Data/labelsB.csv')


iEqualizeMixtureCoeffs = True
target, target_test, source, source_test, targetLabels_train, targetLabels_test, sourceLabels, sourceLabels_test = dh.getCytofMMDDataFromCsv(sample1Path, sample1LabelsPath, sample2Path, sample2LabelsPath, iEqualizeMixtureCoeffs)

# choose all markers
relevantMarkers = range(target.shape[1])
target = target[:,relevantMarkers]
target_test = target_test[:,relevantMarkers]
source = source[:,relevantMarkers] 
source_test = source_test[:,relevantMarkers] 


# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
target_test = dh.preProcessCytofData(target_test)
source = dh.preProcessCytofData(source) 
source_test = dh.preProcessCytofData(source_test) 

target, target_test, source, source_test = dh.standard_scale(target, target_test, source, source_test)
    
input_dim = target.shape[1]

if space == 'code':
    ######################
    ###### train AE ######
    ######################
    
    #create Autoencoder
    encoderLayerSizes = [40,20,12]
    encoding_dim = encoderLayerSizes[-1]
    activation='softplus'
    l2_penalty = 1e-4
    inputLayer = Input(shape=(input_dim,))
    en_hidden1 = Dense(encoderLayerSizes[0], activation=activation,W_regularizer=l2(l2_penalty))(inputLayer)
    en_hidden2 = Dense(encoderLayerSizes[1], activation=activation,W_regularizer=l2(l2_penalty))(en_hidden1)
    code = Dense(encoderLayerSizes[2], activation='sigmoid',W_regularizer=l2(l2_penalty))(en_hidden2)
    de_hidden4 = Dense(encoderLayerSizes[1], activation=activation,W_regularizer=l2(l2_penalty))(code)
    de_hidden5 = Dense(encoderLayerSizes[0], activation=activation,W_regularizer=l2(l2_penalty))(de_hidden4)
    recon = Dense(input_dim, activation='linear',W_regularizer=l2(l2_penalty))(de_hidden5)
    autoencoder = Model(input=inputLayer, output=recon)
    #create encoder
    encoder = Model(input=inputLayer, output=code)
    #create decoder
    encoded_input = Input(shape=(encoding_dim,))
    de_hidden4 = autoencoder.layers[-3](encoded_input)
    de_hidden5 = autoencoder.layers[-2](de_hidden4)
    recon = autoencoder.layers[-1](de_hidden5)
    decoder = Model(input=encoded_input, output=recon)
    #train the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(target, target, nb_epoch=200, batch_size=128, shuffle=True,  validation_split=0.1,
                    callbacks=[mn.monitor(), cb.EarlyStopping(monitor='val_loss', patience=10,  mode='auto')])    
    #Map data to code space
    mmdNetTarget_train = encoder.predict(target)
    mmdNetTarget_test = encoder.predict(target_test)
    mmdNetInput_train=encoder.predict(source)
    mmdNetInput_test=encoder.predict(source_test)
    space_dim = encoding_dim
    # plot reconstructions
    plt.subplot(211)
    plt.scatter(target[:,0], target[:,1], color = 'blue', s=6)
    plt.title("test data")
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    plt.subplot(212)
    x_test_recon = decoder.predict(encoder.predict(target))
    plt.scatter(x_test_recon[:,0], x_test_recon[:,1], color='red', s=6)
    plt.title("test reconstruction")
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    plt.draw()
else: # space = 'data'
    space_dim = input_dim
    mmdNetTarget_train = target
    mmdNetTarget_test = target_test
    mmdNetInput_train = source
    mmdNetInput_test = source_test
    del target, target_test, source, source_test    

################################
######## train MMD-ResNet ######
################################

#Define net configuration
mmdNetLayerSizes = [25, 25, 8]
l2_penalty = 0e-2
l2_penalty2 = 0e-2
init = lambda shape, name:initializations.normal(shape, scale=.1e-4, name=name)
# init = 'glorot_normal' # works if the network is much wider 
calibInput = Input(shape=(space_dim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty2), init = init)(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(space_dim, activation='linear',W_regularizer=l2(l2_penalty2), init = init)(block1_a2) 
block1_output = merge([block1_w2, calibInput], mode = 'sum')
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty2), init = init)(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(space_dim, activation='linear',W_regularizer=l2(l2_penalty2), init = init)(block2_a2) 
block2_output = merge([block2_w2, block1_output], mode = 'sum')

calibMMDNet = Model(input=calibInput, output=block2_output)



# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.1
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

#train MMD net
optimizer = keras.optimizers.adam(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block2_output,mmdNetTarget_train,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

calibMMDNet.fit(mmdNetInput_train,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate,mn.monitorMMD(mmdNetInput_train, mmdNetTarget_train, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=20,mode='auto')])


##############################
###### evaluate results ######
##############################

if space == 'data':
    TargetData = mmdNetTarget_train
    beforeCalib = mmdNetInput_train
    afterCalib = calibMMDNet.predict(mmdNetInput_train)
else:     # space = 'code'
    TargetData = decoder.predict(mmdNetTarget_train)
    beforeCalib = decoder.predict(mmdNetInput_train)
    afterCalib = decoder.predict(calibMMDNet.predict(mmdNetInput_train))

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(TargetData)

# project data onto PCs
target_sample_pca = pca.transform(TargetData)
projection_before = pca.transform(beforeCalib)
projection_after = pca.transform(afterCalib)

# remove cells with unknown cell types
target_sample_pca = target_sample_pca[targetLabels_train!=0]
projection_before = projection_before[sourceLabels!=0]
projection_after = projection_after[sourceLabels!=0]

# choose PCs to plot
pc1 = 0
pc2 = 1
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2])
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after[:,pc1], projection_after[:,pc2])

##################################### qualitative evaluation: per-marker empirical cdfs #####################################

for i in range(len(relevantMarkers)):
    targetMarker = TargetData[:,i]
    beforeMarker = beforeCalib[:,i]
    afterMarker = afterCalib[:,i]
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
corrB = np.corrcoef(beforeCalib, rowvar=0)
corrA = np.corrcoef(afterCalib, rowvar=0)
corrT =     np.corrcoef(TargetData, rowvar=0)
FB = corrB - corrT
FA = corrA - corrT
NB = np.linalg.norm(FB, 'fro')
NA = np.linalg.norm(FA, 'fro')
print('norm before calibration: ', str(NB))
print('norm after calibration: ', str(NA)) 

m1 = np.max(np.abs(FB))
m2 = np.max(np.abs(FA))
m = max(m1, m2)
from  matplotlib.pyplot import cm
fig, ax = plt.subplots()
heatmap = ax.pcolor(FB, cmap=cm.Reds, vmin=-m, vmax=m)
plt.title('before calibration')
fig, ax = plt.subplots()
heatmap = ax.pcolor(FA, cmap=cm.Reds, vmin=-m, vmax=m)
plt.title('after calibration')

##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = beforeCalib.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = TargetData.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(block2_output,TargetData).cost(K.variable(value=beforeCalib[sourceInds]), K.variable(value=TargetData[targetInds])))
mmd_after = K.eval(cf.MMD(block2_output,TargetData).cost(K.variable(value=afterCalib[sourceInds]), K.variable(value=TargetData[targetInds])))
print('MMD before calibration: ' + str(mmd_before))
print('MMD after calibration: ' + str(mmd_after))

