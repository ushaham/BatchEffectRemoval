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
import scipy
from matplotlib import pyplot as plt



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
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a1) 
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(inputDim, activation='linear',W_regularizer=l2(l2_penalty), init = my_init)(block3_a2) 
block3_output = merge([block3_w2, block2_output], mode = 'sum')
ResNet = Model(input=calibInput, output=block3_output)
ResNet.compile(optimizer='rmsprop', loss=lambda y_true,y_pred: 
               cf.MMD(block3_output,target,MMDTargetValidation_split=0.1).KerasCost(y_true,y_pred))

###########################
###### load MMD nets ######
###########################
# we load two CyTOF samples 

if data =='person1_baseline': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  
if data =='person2_baseline': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_baseline_ResNet_weights.h5'))  
if data =='person1_3month': 
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_3month_ResNet_weights.h5'))  
if data =='person2_3month':  
    ResNet.load_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person2_3month_ResNet_weights.h5'))  
   


calibratedSource_resNet = ResNet.predict(source)

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


####################
###### Combat ######
####################
if data =='person1_baseline':
    CombatPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person1_baseline_Combat.csv')
    BatchPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person1_baseline_batch.csv')
if data =='person2_baseline':
    CombatPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person2_baseline_Combat.csv')
    BatchPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person2_baseline_batch.csv')
if data =='person1_3month':
    CombatPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person1_3month_Combat.csv')
    BatchPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person1_3month_batch.csv')
if data =='person2_3month':
    CombatPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person2_3month_Combat.csv')
    BatchPath = os.path.join(io.DeepLearningRoot(),'fromJun/Person2_3month_batch.csv')            

combatData = genfromtxt(CombatPath, delimiter=',', skip_header=0)
combatBatch = genfromtxt(BatchPath, delimiter=',', skip_header=0)
calibratedSource_combat = combatData[combatBatch==1]
calibratedTarget_combat = combatData[combatBatch==2]

calibratedSource_combat = preprocessor.transform(calibratedSource_combat)
calibratedTarget_combat = preprocessor.transform(calibratedTarget_combat)

##############################################
############## Evaluation: MMD ###############
##############################################
# MMD with the scales used for training 
mmd_before = np.zeros(5)
mmd_after_Z = np.zeros(5)
mmd_after_pca = np.zeros(5)
mmd_after_resNet = np.zeros(5)
mmd_after_combat = np.zeros(5)

for i in range(5):
    sourceInds = np.random.randint(low=0, high = source.shape[0], size = 1000)
    targetInds = np.random.randint(low=0, high = target.shape[0], size = 1000)
    mmd_before[i] = K.eval(cf.MMD(source,target).cost(K.variable(value=source[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_Z[i] = K.eval(cf.MMD(calibratedSource_Z,target).cost(K.variable(value=calibratedSource_Z[sourceInds]), K.variable(value=target[targetInds])))
    mmd_after_pca[i] = K.eval(cf.MMD(calibratedSource_pca,calibratedTarget_pca).cost(K.variable(value=calibratedSource_pca[sourceInds]), K.variable(value=calibratedTarget_pca[targetInds])))
    mmd_after_combat[i] = K.eval(cf.MMD(calibratedSource_combat,calibratedTarget_combat).cost(K.variable(value=calibratedSource_combat[sourceInds]), K.variable(value=calibratedTarget_combat[targetInds])))
    mmd_after_resNet[i] = K.eval(cf.MMD(calibratedSource_resNet,target).cost(K.variable(value=calibratedSource_resNet[sourceInds]), K.variable(value=target[targetInds])))


print('MMD before calibration:          ' + str(np.mean(mmd_before))+'pm '+str(np.std(mmd_before)))
print('MMD after calibration (Z):       ' + str(np.mean(mmd_after_Z))+'pm '+str(np.std(mmd_after_Z)))
print('MMD after calibration (PCA):     ' + str(np.mean(mmd_after_pca))+'pm '+str(np.std(mmd_after_pca)))
print('MMD after calibration (Combat):  ' + str(np.mean(mmd_after_combat))+'pm '+str(np.std(mmd_after_combat)))
print('MMD after calibration (resNet):  ' + str(np.mean(mmd_after_resNet))+'pm '+str(np.std(mmd_after_resNet)))


'''
patient 1_baseline:
MMD before calibration:          0.638668644428pm 0.014563096906
MMD after calibration (Z):       0.276640373468pm 0.012924549692
MMD after calibration (PCA):     0.402770000696pm 0.0136063583736
MMD after calibration (Combat):  0.277381533384pm 0.014350908313
MMD after calibration (resNet):  0.191800534725pm 0.00762761862609

patient 2_baseline:
MMD before calibration:          0.555728244781pm 0.00789111145345
MMD after calibration (Z):       0.265854990482pm 0.0155749355254
MMD after calibration (PCA):     0.401724106073pm 0.0125636161256
MMD after calibration (Combat):  0.255779594183pm 0.0045216242106
MMD after calibration (resNet):  0.164374938607pm 0.00615682418355

patient 1_3month:
MMD before calibration:          0.623963487148pm 0.0187013509273
MMD after calibration (Z):       0.292916560173pm 0.015575261872
MMD after calibration (PCA):     0.359512370825pm 0.0111780519185
MMD after calibration (Combat):  0.297321844101pm 0.0142308482008
MMD after calibration (resNet):  0.194353529811pm 0.0104502178413

patient 2_3month:
MMD before calibration:          0.659076714516pm 0.00553641512917
MMD after calibration (Z):       0.296549755335pm 0.00929416024482
MMD after calibration (PCA):     0.367381608486pm 0.0094637884607
MMD after calibration (Combat):  0.293218165636pm 0.00663473122269
MMD after calibration (resNet):  0.197239124775pm 0.00868047602136


'''




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
target_after_combat = pca.transform(calibratedTarget_combat)
projection_after_combat = pca.transform(calibratedSource_combat)
projection_after_net = pca.transform(calibratedSource_resNet)

# no calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_before[:,pc1], projection_before[:,pc2])
# z transform calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_z[:,pc1], projection_after_z[:,pc2])
# PCA calibration
sh.scatterHist(target_after_pca[:,pc1], target_after_pca[:,pc2], projection_after_pca[:,pc1], projection_after_pca[:,pc2])
# Combat calibration
sh.scatterHist(target_after_combat[:,pc1], target_after_combat[:,pc2], projection_after_combat[:,pc1], projection_after_combat[:,pc2])
#ResNet calibration
sh.scatterHist(target_sample_pca[:,pc1], target_sample_pca[:,pc2], projection_after_net[:,pc1], projection_after_net[:,pc2])

sourceLabels = genfromtxt(sourceLabelPath, delimiter=',', skip_header=0)
targetLabels = genfromtxt(targetLabelPath, delimiter=',', skip_header=0)

source_subPop = source[sourceLabels==1]
resNetCalibSubPop = calibratedSource_resNet[sourceLabels==1]
Z_CalibSubPop = calibratedSource_Z[sourceLabels==1]
pca_CalibSubPop = calibratedSource_pca[sourceLabels==1]
target_subPop = target[targetLabels==1]

marker1 = 1 #17 'IFNg'
marker2 = 18

axis1 = 'CD28'
axis2 = 'GZB'

# CD8 population plots
# before calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], source_subPop[:,marker1], source_subPop[:,marker2], axis1, axis2)
#Z transform calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], Z_CalibSubPop[:,marker1], Z_CalibSubPop[:,marker2], axis1, axis2)
# PCA calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], pca_CalibSubPop[:,marker1], pca_CalibSubPop[:,marker2], axis1, axis2)
# ResNet calibration
sh.scatterHist(target_subPop[:,marker1], target_subPop[:,marker2], resNetCalibSubPop[:,marker1], resNetCalibSubPop[:,marker2], axis1, axis2)





# KS test
d = source.shape[1]
pVals = np.zeros((d,5))
for i in range(d):
    pVals[i,0] = scipy.stats.ks_2samp(source[:,i], target[:,i])[1]
    pVals[i,1] = scipy.stats.ks_2samp(calibratedSource_Z[:,i], target[:,i])[1]
    pVals[i,2] = scipy.stats.ks_2samp(calibratedSource_pca[:,i], calibratedTarget_pca[:,i])[1]
    pVals[i,3] = scipy.stats.ks_2samp(calibratedSource_combat[:,i], calibratedTarget_combat[:,i])[1]
    pVals[i,4] = scipy.stats.ks_2samp(calibratedSource_resNet[:,i], target[:,i])[1]
fig, (a1)  = plt.subplots(1,1)
a1.hist(pVals, normed=True) 
#plt.legend(['no calibration', 'mean, var maching','PCA', 'MMD-ResNet'],prop={'size':16})
plt.legend(['no calibration', 'mean, var maching','PCA', 'Combat', 'MMD-ResNet'])
a1.axes.get_yaxis().set_visible(False)
plt.show()  
    
 