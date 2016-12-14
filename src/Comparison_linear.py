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
from matplotlib import pyplot as plt
import CostFunctions as cf
from sklearn import decomposition
from keras import backend as K
import ScatterHist as sh
from statsmodels.distributions.empirical_distribution import ECDF
from numpy import genfromtxt
import sklearn.preprocessing as prep
from keras.models import load_model
from keras import initializations
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, merge, Activation
from keras.regularizers import l2
from keras.models import Model



# configuration hyper parameters
denoise = True # whether or not to use a denoising autoencoder to remove the zeros

######################
###### get data ######
######################
# we load two CyTOF samples 

data = 'person1_baseline'

if data =='person1_baseline':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_baseline.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_baseline.csv')
    sourceLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_baseline_label.csv')
    targetLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_baseline_label.csv')
    autoencoder =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_DAE.h5'))   
if data =='person2_baseline':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_baseline.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_baseline.csv')
    sourceLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_baseline_label.csv')
    targetLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_baseline_label.csv')
    autoencoder =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_DAE.h5'))  
if data =='person1_3month':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_3month.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_3month.csv')
    sourceLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_3month_label.csv')
    targetLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_3month_label.csv')
    autoencoder =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_DAE.h5'))    
if data =='person2_3month':
    sourcePath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_3month.csv')
    targetPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_3month.csv')
    sourceLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_3month_label.csv')
    targetLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_3month_label.csv')
    autoencoder =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_DAE.h5'))   
   
source = genfromtxt(sourcePath, delimiter=',', skip_header=0)
target = genfromtxt(targetPath, delimiter=',', skip_header=0)


# pre-process data: log transformation, a standard practice with CyTOF data
target = dh.preProcessCytofData(target)
source = dh.preProcessCytofData(source) 

if denoise:
    source = autoencoder.predict(source)
    target = autoencoder.predict(target)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source) 
target = preprocessor.transform(target)    




###########################
###### define models ######
###########################
# we load two CyTOF samples 
mmdNetLayerSizes = [25, 25]
inputDim = 25
l2_penalty = 1e-2

def my_init (shape, name = None):
    return initializations.normal(shape, scale=.1e-4, name=name)
setattr(initializations, 'my_init', my_init)


# resNet
calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1) 
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2) 
block1_output = merge([block1_w2, calibInput], mode = 'sum')
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1) 
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2) 
block2_output = merge([block2_w2, block1_output], mode = 'sum')
ResNet = Model(input=calibInput, output=block2_output)
ResNet.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block2_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

# MLP
calibInput_mlp = Input(shape=(inputDim,))
block1_bn1_mlp = BatchNormalization()(calibInput_mlp)
block1_a1_mlp = Activation('relu')(block1_bn1_mlp)
block1_w1_mlp = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1_mlp) 
block1_bn2_mlp = BatchNormalization()(block1_w1_mlp)
block1_a2_mlp = Activation('relu')(block1_bn2_mlp)
block1_w2_mlp = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2_mlp) 
block2_bn1_mlp = BatchNormalization()(block1_w2_mlp)
block2_a1_mlp = Activation('relu')(block2_bn1_mlp)
block2_w1_mlp = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1_mlp) 
block2_bn2_mlp = BatchNormalization()(block2_w1_mlp)
block2_a2_mlp = Activation('relu')(block2_bn2_mlp)
block2_w2_mlp = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2_mlp) 
MLP = Model(input=calibInput_mlp, output=block2_w2_mlp)
MLP.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block2_w2_mlp,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

###########################
###### load MMD nets ######
###########################
# we load two CyTOF samples 

data = 'person1_baseline'

if data =='person1_baseline': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  
    MLP.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_MLP_weights.h5'))  
if data =='person2_baseline': 
    ResNet =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_ResNet.h5'))  
    MLP =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_MLP.h5'))  
if data =='person1_3month': 
    ResNet =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_ResNet.h5'))  
    MLP =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_MLP.h5'))  
if data =='person2_3month':  
    ResNet =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_ResNet.h5'))  
    MLP =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_MLP.h5'))  
   


calibratedSource_resNet = ResNet.predict(source)
calibratedSource_MLP = MLP.predict(source)

###############################################################
###### calibration by shifting and rescaling each marker ######
###############################################################

print('linear calibration by matching means and variances')
mt = np.mean(target, axis = 0)
vt = np.var(target, axis = 0)
ms = np.mean(source, axis = 0)
vs = np.var(source, axis = 0)

calibratedSource_Z = np.zeros(source.shape)
dim = target.shape[1]
for i in range(dim):
    calibratedSource_Z[:,i] = (source[:,i] - ms[i]) / np.sqrt(vs[i]) * np.sqrt(vt[i]) + mt[i]
    

##########################################
###### removing principal component ######
##########################################

n_target = target.shape[0]
n_source = source.shape[0]
batch = np.concatenate([np.ones(n_target), 2*np.ones(n_source)], axis=0)
combinedData = np.concatenate([target, source], axis=0)
pca = decomposition.PCA()
pca.fit(combinedData)
combinedData_proj = pca.transform(combinedData)
# measure correlation between each PC and the batch
dim = target.shape[1]
corrs = np.zeros(dim)
for i in range(dim):
    corrs[i] = np.corrcoef(batch, combinedData_proj[:,i])[0,1]
print('correlation coefficients: ')
print(corrs)
toTake = np.argsort(np.abs(corrs))[range(dim-1)] 


combinedDataRecon = np.dot(combinedData_proj[:,toTake], pca.components_[[toTake]]) + np.mean(combinedData, axis=0)

calibratedTarget_pca = combinedDataRecon[:n_target,]
calibratedSource_pca = combinedDataRecon[n_target:,]
# MMD with the a single scale at a time, for various scales

##############################################
############## Evaluation: MMD ###############
##############################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(source,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
mmd_after_Z = K.eval(cf.MMD(calibratedSource_Z,target).cost(K.variable(value=calibratedSource_Z[sourceInds]), K.variable(value=target[targetInds])))
mmd_after_pca = K.eval(cf.MMD(calibratedSource_pca,calibratedTarget_pca).cost(K.variable(value=calibratedSource_pca[sourceInds]), K.variable(value=calibratedTarget_pca[targetInds])))
mmd_after_resNet = K.eval(cf.MMD(calibratedSource_resNet,target).cost(K.variable(value=calibratedSource_resNet[sourceInds]), K.variable(value=target[targetInds])))

print('MMD before calibration: ' + str(mmd_before))
print('MMD after calibration (Z): ' + str(mmd_after_Z))
print('MMD after calibration (PCA): ' + str(mmd_after_pca))
print('MMD after calibration (resNet): ' + str(mmd_after_resNet))


# MMD(target,target):                                                                         [ 0.04440385  0.04514616  0.04236437  0.02476904]
# calibration using matching of means and variances: MMD(after calibration, target):          [ 0.05712092  0.0715281   0.10891724  0.03884947]
# calibration by removing PC most correlated with the batch: MMD(after calibration, target):  [ 0.05697488  0.09389935  0.19798119  0.11305808]
# MMD(after calibration, target):                                                             [ 0.05670956  0.06138352  0.05434766  0.02735295]





##############################################
############## Evaluation: PCA ###############
##############################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
pc1 = 1
pc2 = 2

projection_after_z = pca.transform(calibratedSource_Z)
target_after_pca = pca.transform(calibratedTarget_pca)
projection_after_pca = pca.transform(calibratedSource_pca)
projection_after_net = pca.transform(calibratedSource_resNet)


sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2])
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_z[:,pc1], projection_after_z[:,pc2])
sh.scatterHist(target_after_pca[:,pc1], target_after_pca[:,pc2], projection_after_pca[:,pc1], projection_after_pca[:,pc2])
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_net[:,pc1], projection_after_net[:,pc2])
