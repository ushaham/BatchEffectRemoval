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

data = 'person2_baseline'

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

if data =='person1_baseline': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  
    MLP.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_MLP_weights.h5'))  
if data =='person2_baseline': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_ResNet_weights.h5'))  
    MLP.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_MLP_weights.h5'))  
if data =='person1_3month': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_ResNet_weights.h5'))  
    MLP.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_MLP_weights.h5'))  
if data =='person2_3month':  
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_ResNet_weights.h5'))  
    MLP.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_MLP_weights.h5'))  
   

##############################
###### evaluate results ######
##############################

calibratedSource_resNet = ResNet.predict(source)
calibratedSource_MLP = MLP.predict(source)

##################################### qualitative evaluation: PCA #####################################
pca = decomposition.PCA()
pca.fit(target)

# project data onto PCs
target_sample_pca = pca.transform(target)
projection_before = pca.transform(source)
projection_after_ResNet = pca.transform(calibratedSource_resNet)
projection_after_MLP = pca.transform(calibratedSource_MLP)
 
# choose PCs to plot
pc1 = 0
pc2 = 1
axis1 = 'PC'+str(pc1)
axis2 = 'PC'+str(pc2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_ResNet[:,pc1], projection_after_ResNet[:,pc2], axis1, axis2)
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_MLP[:,pc1], projection_after_MLP[:,pc2], axis1, axis2)

##################################### qualitative evaluation: per-marker empirical cdfs #####################################
# plot a few markers before and after calibration
for i in range(np.min([10,target.shape[1]])):
    targetMarker = target[:,i]
    beforeMarker = source[:,i]
    afterMarker = calibratedSource_resNet[:,i]
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
# compute the correlation matrices C_source, C_target before and after calibration 
# and plot a histogram of the values of C_diff = C_source-C_target
corrB = np.corrcoef(source, rowvar=0)
corrA_resNet = np.corrcoef(calibratedSource_resNet, rowvar=0)
corrA_MLP = np.corrcoef(calibratedSource_MLP, rowvar=0)

corrT = np.corrcoef(target, rowvar=0)
FB = corrT - corrB
FA_resNet = corrT - corrA_resNet
FA_MLP= corrT - corrA_MLP

NB = np.linalg.norm(FB, 'fro')
NA_resNet = np.linalg.norm(FA_resNet, 'fro')
NA_MLP = np.linalg.norm(FA_MLP, 'fro')


print('norm before calibration:         ', str(NB))
print('norm after calibration (resNet): ', str(NA_resNet)) 
print('norm after calibration (MLP):    ', str(NA_MLP)) 


'''
patient 1_baseline:
norm before calibration:          3.13282024726
norm after calibration (resNet):  2.54434524732
norm after calibration (MLP):     3.84227336367

patient 2_baseline:
norm before calibration:          2.52701517319
norm after calibration (resNet):  1.56590447629
norm after calibration (MLP):     2.10564254635

patient 1_3month:
norm before calibration:          1.33466077191
norm after calibration (resNet):  2.82159111885
norm after calibration (MLP):     4.61930150125

patient 2_3month:
norm before calibration:          2.05579245152
norm after calibration (resNet):  3.45321065574
norm after calibration (MLP):     3.4712589682
'''

fa_resNet = FA_resNet.flatten()
fa_MLP = FA_MLP.flatten()
fb = FB.flatten()

f = np.zeros((fa_resNet.shape[0],3))
f[:,0] = fb
f[:,1] = fa_resNet
f[:,2] = fa_MLP

fig = plt.figure()
plt.hist(f, bins = 20, normed=True, histtype='bar')
plt.legend(['before calib.', 'ResNet calib.', 'MLP calib.'], loc=1)
plt.yticks([])
plt.show()
##################################### quantitative evaluation: MMD #####################################
# MMD with the scales used for training 

sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)

mmd_before = K.eval(cf.MMD(source,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
mmd_after_resNet = K.eval(cf.MMD(calibratedSource_resNet,target).cost(K.variable(value=calibratedSource_resNet[sourceInds]), K.variable(value=target[targetInds])))
mmd_after_MLP = K.eval(cf.MMD(calibratedSource_MLP,target).cost(K.variable(value=calibratedSource_MLP[sourceInds]), K.variable(value=target[targetInds])))

print('MMD before calibration:         ' + str(mmd_before))
print('MMD after calibration (resNet): ' + str(mmd_after_resNet))
print('MMD after calibration (MLP):    ' + str(mmd_after_MLP))

'''
patient 1_baseline:
MMD before calibration:         0.679205
MMD after calibration (resNet): 0.291898
MMD after calibration (MLP):    0.362767

patient 2_baseline:
MMD before calibration:         0.618438
MMD after calibration (resNet): 0.275275
MMD after calibration (MLP):    0.34958

patient 1_3month:
MMD before calibration:         0.731892
MMD after calibration (resNet): 0.381757
MMD after calibration (MLP):    0.350164

patient 2_3month:
MMD before calibration:         0.691358
MMD after calibration (resNet): 0.305009
MMD after calibration (MLP):    0.382219

'''

##################################### CD8 sub-population #####################################
sourceLabels = genfromtxt(sourceLabelPath, delimiter=',', skip_header=0)
targetLabels = genfromtxt(targetLabelPath, delimiter=',', skip_header=0)

source_subPop = source[sourceLabels==1]
resNetCalibSubPop = calibratedSource_resNet[sourceLabels==1]
mlpCalibSubPop = calibratedSource_MLP[sourceLabels==1]
target_subPop = target[targetLabels==1]

marker1 = 13 #17 'IFNg'
marker2 = 19

axis1 = 'CD28'
axis2 = 'GZB'

sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], source_subPop[:,marker1], source_subPop[:,marker2], axis1, axis2)
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], resNetCalibSubPop[:,marker1], resNetCalibSubPop[:,marker2], axis1, axis2)
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], mlpCalibSubPop[:,marker1], mlpCalibSubPop[:,marker2], axis1, axis2)
