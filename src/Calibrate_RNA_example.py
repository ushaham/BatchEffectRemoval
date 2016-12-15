'''
Created on Sep 13, 2016

@author: Uri Shaham
''' 

import os.path
from Calibration_Util import FileIO as io
import keras.optimizers
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
from keras.callbacks import LearningRateScheduler
import math
from keras import backend as K
from numpy import genfromtxt
from numpy import savetxt
import sklearn.preprocessing as prep
from sklearn import decomposition
import ScatterHist as sh
from statsmodels.distributions.empirical_distribution import ECDF
from keras import initializations 

# the input here is standardized PCs

#Define net configuration
mmdNetLayerSizes = [50,50,50]
l2_penalty = 1e-2
#my_init = 'glorot_normal' 
def my_init (shape, name = None):
    return initializations.normal(shape, scale=.1e-4, name=name)
    

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = epochsDrop
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
lrate = LearningRateScheduler(step_decay)

patience = 50
epochsDrop = 50.0

dataPath = '/raid3/DropSeq/Retina/Second/1215/Data2_standardized_37PCs.csv'    
#dataPath = os.path.join(io.DeepLearningRoot(),'Data/Data2_standardized_37PCs.csv')
data = genfromtxt(dataPath, delimiter=',', skip_header=0)
    
batchesPath = '/raid3/DropSeq/Retina/Second/1215/batch.csv'  
#batchesPath = os.path.join(io.DeepLearningRoot(),'Data/batch.csv')
batches = genfromtxt(batchesPath, delimiter=',', skip_header=0) 

targetBatch = 1
target = data[batches == targetBatch]
# standardize

pca = decomposition.PCA()
pca.fit(data) 
# project data onto PCs
target_sample_pca = pca.transform(target)
space_dim = target.shape[1]
calibratedData = np.zeros((0, space_dim))
batch_res = np.zeros((0, 1))
calibratedData = np.concatenate([calibratedData, target], axis=0)
batch_res = np.concatenate([batch_res, targetBatch*np.ones((target.shape[0],1))], axis=0)

sourceBatch = 2
# load data
source = data[batches == sourceBatch]
# define net    
calibInput = Input(shape=(space_dim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(space_dim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2) 
block1_output = merge([block1_w2, calibInput], mode = 'sum')
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(space_dim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2) 
block2_output = merge([block2_w2, block1_output], mode = 'sum')
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1) 
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(space_dim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2) 
block3_output = merge([block3_w2, block2_output], mode = 'sum')

calibMMDNet = Model(input=calibInput, output=block3_output)


#train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true,y_pred: 
               cf.MMD(block3_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))
sourceLabels = np.zeros(source.shape[0])
calibMMDNet.fit(source,sourceLabels,nb_epoch=500,batch_size=1000,validation_split=0.1,verbose=1,
           callbacks=[lrate, mn.monitorMMD(source, target, calibMMDNet.predict),
                      cb.EarlyStopping(monitor='val_loss',patience=50,mode='auto')])

afterCalib = calibMMDNet.predict(source)

##################################### qualitative evaluation: PCA #####################################
projection_before = pca.transform(source)
projection_after = pca.transform(afterCalib)    
# The PCs most correlated with the batch are 3 and 5
pc1 = 3
pc2 = 5

sh.scatterHist(target[:,pc1], target[:,pc2], source[:,pc1], source[:,pc2])
sh.scatterHist(target[:,pc1], target[:,pc2], afterCalib[:,pc1], afterCalib[:,pc2])

        
##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(block2_output,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
mmd_after = K.eval(cf.MMD(block2_output,target).cost(K.variable(value=afterCalib[sourceInds]), K.variable(value=target[targetInds])))
print('MMD before calibration: ' + str(mmd_before))
print('MMD after calibration: ' + str(mmd_after))

'''
this script gives:
MMD before calibration: 0.384037
MMD after calibration: 0.142719
'''
     
############################ save results ######################################## 
calibratedSource =  calibMMDNet.predict(source)
calibratedData = np.concatenate([calibratedData, calibratedSource], axis=0)
'''
# save calibrated data
cal_filename = '/raid3/uri/RNA_second_calibratedData1215.csv'    
savetxt(cal_filename, calibratedData, delimiter = ",")
 
# save model
calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),'savedModels/RNA_ResNet_weights.h5'))  
'''