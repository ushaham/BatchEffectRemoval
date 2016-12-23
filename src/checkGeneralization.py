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
from sklearn import decomposition
from keras import backend as K
import ScatterHist as sh
from numpy import genfromtxt
import sklearn.preprocessing as prep
from keras.models import load_model
from keras import initializations
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, merge, Activation
from keras.regularizers import l2
from keras.models import Model


# configuration hyper parameters
denoise = True # whether or not to train a denoising autoencoder to remover the zeros

######################
###### get data ######
######################
# we load two CyTOF samples 

p1d1Path = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_baseline.csv')
p1d2Path = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_baseline.csv')
p2d1Path = os.path.join(io.DeepLearningRoot(),'Data/Person2Day1_baseline.csv')
p2d2Path = os.path.join(io.DeepLearningRoot(),'Data/Person2Day2_baseline.csv')

p1Dae =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_DAE.h5'))  
p2Dae =  load_model(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_DAE.h5'))  

   
p1d1 = genfromtxt(p1d1Path, delimiter=',', skip_header=0)
p1d2 = genfromtxt(p1d2Path, delimiter=',', skip_header=0)
p2d1 = genfromtxt(p2d1Path, delimiter=',', skip_header=0)
p2d2 = genfromtxt(p2d2Path, delimiter=',', skip_header=0)



# pre-process data: log transformation, a standard practice with CyTOF data
p1d1 = dh.preProcessCytofData(p1d1)
p1d2 = dh.preProcessCytofData(p1d2) 
p2d1 = dh.preProcessCytofData(p2d1)
p2d2 = dh.preProcessCytofData(p2d2) 

if denoise:
    p1d1 = p1Dae.predict(p1d1)
    p1d2 = p1Dae.predict(p1d2)
    p2d1 = p2Dae.predict(p2d1)
    p2d2 = p2Dae.predict(p2d2)

# rescale source to have zero mean and unit variance
# apply same transformation to the target
p1d1_pp = prep.StandardScaler().fit(p1d1)
p1d2_pp = prep.StandardScaler().fit(p1d2)
p2d1_pp = prep.StandardScaler().fit(p2d1)
p2d2_pp = prep.StandardScaler().fit(p2d2)

'''
net1 maps p1d1 to p2d1 (Nd1)
net2 maps p2d1 to p2d2 (Np2)
net3 maps p2d2 to p1d2 (Nd2)
net4 maps p1d1 to p1d2 (Nd1)
'''
net1_target = p1d1_pp.transform(p2d1) 
net2_target = p2d1_pp.transform(p2d2)
net3_target = p1d1_pp.transform(p1d2)
net4_target = p1d1_pp.transform(p1d2) 

net1_source = p1d1_pp.transform(p1d1) 
net4_source = p1d1_pp.transform(p1d1) 

##############################
######## load ResNets ########
##############################
mmdNetLayerSizes = [25, 25]
inputDim = 25
l2_penalty = 1e-2

def my_init (shape, name = None):
    return initializations.normal(shape, scale=.1e-4, name=name)
setattr(initializations, 'my_init', my_init)

calibInput_1 = Input(shape=(inputDim,))
block1_bn1_1 = BatchNormalization()(calibInput_1)
block1_a1_1 = Activation('relu')(block1_bn1_1)
block1_w1_1 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1_1) 
block1_bn2_1 = BatchNormalization()(block1_w1_1)
block1_a2_1 = Activation('relu')(block1_bn2_1)
block1_w2_1 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2_1) 
block1_output_1 = merge([block1_w2_1, calibInput_1], mode = 'sum')
block2_bn1_1 = BatchNormalization()(block1_output_1)
block2_a1_1 = Activation('relu')(block2_bn1_1)
block2_w1_1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1_1) 
block2_bn2_1 = BatchNormalization()(block2_w1_1)
block2_a2_1 = Activation('relu')(block2_bn2_1)
block2_w2_1 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2_1) 
block2_output_1 = merge([block2_w2_1, block1_output_1], mode = 'sum')
block3_bn1_1 = BatchNormalization()(block2_output_1)
block3_a1_1 = Activation('relu')(block3_bn1_1)
block3_w1_1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1_1) 
block3_bn2_1 = BatchNormalization()(block3_w1_1)
block3_a2_1 = Activation('relu')(block3_bn2_1)
block3_w2_1 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2_1) 
block3_output_1 = merge([block3_w2_1, block2_output_1], mode = 'sum')
ResNet1 = Model(input=calibInput_1, output=block3_output_1)
ResNet1.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block3_output_1,net1_target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

calibInput_2 = Input(shape=(inputDim,))
block1_bn1_2 = BatchNormalization()(calibInput_2)
block1_a1_2 = Activation('relu')(block1_bn1_2)
block1_w1_2 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1_2) 
block1_bn2_2 = BatchNormalization()(block1_w1_2)
block1_a2_2 = Activation('relu')(block1_bn2_2)
block1_w2_2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2_2) 
block1_output_2 = merge([block1_w2_2, calibInput_2], mode = 'sum')
block2_bn1_2 = BatchNormalization()(block1_output_2)
block2_a1_2 = Activation('relu')(block2_bn1_2)
block2_w1_2 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1_2) 
block2_bn2_2 = BatchNormalization()(block2_w1_2)
block2_a2_2 = Activation('relu')(block2_bn2_2)
block2_w2_2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2_2) 
block2_output_2 = merge([block2_w2_2, block1_output_2], mode = 'sum')
block3_bn1_2 = BatchNormalization()(block2_output_2)
block3_a1_2 = Activation('relu')(block3_bn1_2)
block3_w1_2 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1_2) 
block3_bn2_2 = BatchNormalization()(block3_w1_2)
block3_a2_2 = Activation('relu')(block3_bn2_2)
block3_w2_2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2_2) 
block3_output_2 = merge([block3_w2_2, block2_output_2], mode = 'sum')
ResNet2 = Model(input=calibInput_2, output=block3_output_2)
ResNet2.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block3_output_2,net2_target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

calibInput_3 = Input(shape=(inputDim,))
block1_bn1_3 = BatchNormalization()(calibInput_3)
block1_a1_3 = Activation('relu')(block1_bn1_3)
block1_w1_3 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1_3) 
block1_bn2_3 = BatchNormalization()(block1_w1_3)
block1_a2_3 = Activation('relu')(block1_bn2_3)
block1_w2_3 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2_3) 
block1_output_3 = merge([block1_w2_3, calibInput_3], mode = 'sum')
block2_bn1_3 = BatchNormalization()(block1_output_3)
block2_a1_3 = Activation('relu')(block2_bn1_3)
block2_w1_3 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1_3) 
block2_bn2_3 = BatchNormalization()(block2_w1_3)
block2_a2_3 = Activation('relu')(block2_bn2_3)
block2_w2_3 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2_3) 
block2_output_3 = merge([block2_w2_3, block1_output_3], mode = 'sum')
block3_bn1_3 = BatchNormalization()(block2_output_3)
block3_a1_3 = Activation('relu')(block3_bn1_3)
block3_w1_3 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1_3) 
block3_bn2_3 = BatchNormalization()(block3_w1_3)
block3_a2_3 = Activation('relu')(block3_bn2_3)
block3_w2_3 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2_3) 
block3_output_3 = merge([block3_w2_3, block2_output_3], mode = 'sum')
ResNet3 = Model(input=calibInput_3, output=block3_output_3)
ResNet3.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block3_output_3,net3_target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

calibInput_4 = Input(shape=(inputDim,))
block1_bn1_4 = BatchNormalization()(calibInput_4)
block1_a1_4 = Activation('relu')(block1_bn1_4)
block1_w1_4 = Dense(mmdNetLayerSizes[0], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a1_4) 
block1_bn2_4 = BatchNormalization()(block1_w1_4)
block1_a2_4 = Activation('relu')(block1_bn2_4)
block1_w2_4 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block1_a2_4) 
block1_output_4 = merge([block1_w2_4, calibInput_4], mode = 'sum')
block2_bn1_4 = BatchNormalization()(block1_output_4)
block2_a1_4 = Activation('relu')(block2_bn1_4)
block2_w1_4 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a1_4) 
block2_bn2_4 = BatchNormalization()(block2_w1_4)
block2_a2_4 = Activation('relu')(block2_bn2_4)
block2_w2_4 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block2_a2_4) 
block2_output_4 = merge([block2_w2_4, block1_output_4], mode = 'sum')
block3_bn1_4 = BatchNormalization()(block2_output_4)
block3_a1_4 = Activation('relu')(block3_bn1_4)
block3_w1_4 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1_4) 
block3_bn2_4 = BatchNormalization()(block3_w1_4)
block3_a2_4 = Activation('relu')(block3_bn2_4)
block3_w2_4 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2_4) 
block3_output_4 = merge([block3_w2_4, block2_output_4], mode = 'sum')
ResNet4 = Model(input=calibInput_4, output=block3_output_4)
ResNet4.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block3_output_4,net4_target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))





ResNet1.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_2_baseline_weights.h5'))  
ResNet2.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_ResNet_weights.h5'))  
ResNet3.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_1_baseline_weights.h5'))  
ResNet4.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  

##################################
######### compute the map ########
##################################
net1Output = ResNet1.predict(net1_source)
net1Output_adj = p1d1_pp.inverse_transform(net1Output)
net2Input = p2d1_pp.transform(net1Output_adj)
net2Output = ResNet2.predict(net2Input)
net2Output_adj = p2d1_pp.inverse_transform(net2Output)
net3Input = p2d2_pp.transform(net2Output_adj)
net3Output = ResNet3.predict(net3Input)
net3Output_adj = p2d2_pp.inverse_transform(net3Output)
net1_3Calib = p1d1_pp.transform(net3Output_adj)
net4Calib = ResNet4.predict(net4_source)


##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(net4_target)

# project data onto PCs
target_sample_pca = pca.transform(net4_target)
projection_before = pca.transform(net4_source)
projection_short = pca.transform(net4Calib)
projection_long= pca.transform(net1_3Calib)

pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1+1)
axis2 = 'PC'+str(pc2+1)
# before calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
# direct calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_short[:,pc1], projection_short[:,pc2], axis1, axis2)
# indirect calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_long[:,pc1], projection_long[:,pc2], axis1, axis2)



##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 
sourceInds = np.random.randint(low=0, high = p1d1.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = p1d2.shape[0], size = 1000)

mmd_before = np.zeros(5)
mmd_after_short = np.zeros(5)
mmd_after_long = np.zeros(5)

for i in range(5):
    mmd_before[i] = K.eval(cf.MMD(net4_source,net4_target).cost(K.variable(value=net4_source[sourceInds]), K.variable(value=net4_target[targetInds])))
    mmd_after_short[i] = K.eval(cf.MMD(net4Calib,net4_target).cost(K.variable(value=net4Calib[sourceInds]), K.variable(value=net4_target[targetInds])))
    mmd_after_long[i] = K.eval(cf.MMD(net1_3Calib,net4_target).cost(K.variable(value=net1_3Calib[sourceInds]), K.variable(value=net4_target[targetInds])))


print('patient 1: MMD to p1d2 before calibration:             ' + str(np.mean(mmd_before))+'pm '+str(np.std(mmd_before)))
print('patient 1: MMD to p1d2 after calibration (short path): ' + str(np.mean(mmd_after_short))+'pm '+str(np.std(mmd_after_short)))
print('patient 1: MMD to p1d2 after calibration (long path):  ' + str(np.mean(mmd_after_long))+'pm '+str(np.std(mmd_after_long)))

'''
patient 1: MMD to p1d2 before calibration:             0.691462516785pm 0.00192805712724
patient 1: MMD to p1d2 after calibration (short path): 0.277334314585pm 0.000277815282248
patient 1: MMD to p1d2 after calibration (long path):  0.298864048719pm 0.00012550726952
'''

##################################### CD8 sub-population #####################################
sourceLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day1_baseline_label.csv')
targetLabelPath = os.path.join(io.DeepLearningRoot(),'Data/Person1Day2_baseline_label.csv')
sourceLabels = genfromtxt(sourceLabelPath, delimiter=',', skip_header=0)
targetLabels = genfromtxt(targetLabelPath, delimiter=',', skip_header=0)

source_subPop = net4_source[sourceLabels==1]
net4CalibSubPop = net4Calib[sourceLabels==1]
net1_3CalibSubPop = net1_3Calib[sourceLabels==1]
target_subPop = net4_target[targetLabels==1]

marker1 = 13 #17 'IFNg'
marker2 = 19

axis1 = 'CD28'
axis2 = 'GzB'

# before calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], source_subPop[:,marker1], source_subPop[:,marker2], axis1, axis2)
# after direct calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], net4CalibSubPop[:,marker1], net4CalibSubPop[:,marker2], axis1, axis2)
# after indirect calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], net1_3CalibSubPop[:,marker1], net1_3CalibSubPop[:,marker2], axis1, axis2)
