'''
Created on Jul 28, 2016

@author: urishaham
'''


import numpy as np
from numpy import genfromtxt
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as prep


class Sample:
    X = None
    y = None
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

def preProcessCytofData(data):
    return np.log(1+data)

def preProcessSamplesCytofData(samples):
    for i in range(len(samples)):    
        samples[i].X = preProcessCytofData(samples[i].X)
        #s = Sample(preProcessCytofData(samples[i].X), samples[i].y)
    return samples  

    
def getCytofMMDDataFromCsv(sample1Path, sample1LabelsPath, sample2Path, sample2LabelsPath, iEqualizeMixtureCoeffs): 
    print('loading data') 
    sample1 = genfromtxt(sample1Path, delimiter=',', skip_header=0)
    sample2 = genfromtxt(sample2Path, delimiter=',', skip_header=0) 
    sample1Labels = genfromtxt(sample1LabelsPath, delimiter=',').astype(int)
    sample2Labels = genfromtxt(sample2LabelsPath, delimiter=',').astype(int)
    target, target_test, targetLabels_train, targetLabels_test = train_test_split(sample1, 
                                                                            sample1Labels, test_size=0.3) 
    source, source_test, sourceLabels, sourceLabels_test = train_test_split(sample2, 
                                                                            sample2Labels, test_size=0.3)  
    if iEqualizeMixtureCoeffs:
        # we sample with replacement new source samples from the current source samples, according to the target mixtures
        numUniqueLabels = len(set(targetLabels_train))
        
        hist_train = np.histogram(targetLabels_train, bins = numUniqueLabels,density = False)[0] / len(targetLabels_train)        
        sourceInds_train = np.zeros(0)
        n_training = source.shape[0]
        a_train = np.arange(n_training)
        for i in range(numUniqueLabels):
            n_required = int(n_training*hist_train[i])
            labelInds_train = a_train[sourceLabels==i]
            S = np.random.choice(labelInds_train, size=n_required, replace=True)
            sourceInds_train = np.concatenate((sourceInds_train, S), axis=0)
        sourceInds_train = sourceInds_train.astype(int)
        source = source[sourceInds_train]
        sourceLabels = sourceLabels[sourceInds_train]
        
        hist_test = np.histogram(targetLabels_test, bins = numUniqueLabels,density = False)[0] / len(targetLabels_test)        
        sourceInds_test = np.zeros(0)
        n_test = source_test.shape[0]
        a_test = np.arange(n_test)
        for i in range(numUniqueLabels):
            n_required = int(n_test*hist_test[i])
            labelInds_test = a_test[sourceLabels_test==i]
            S = np.random.choice(labelInds_test, size=n_required, replace=True)
            sourceInds_test = np.concatenate((sourceInds_test, S), axis=0)
        sourceInds_test = sourceInds_test.astype(int)
        source_test = source_test[sourceInds_test]
        sourceLabels_test = sourceLabels_test[sourceInds_test]

    return target, target_test, source, source_test, targetLabels_train, targetLabels_test, sourceLabels, sourceLabels_test

def standard_scale(X_train, X_test, X_trainCalib, X_testCalib):
    preprocessor = prep.StandardScaler().fit(X_trainCalib)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    #preprocessor = prep.StandardScaler().fit(X_trainCalib)
    X_trainCalib = preprocessor.transform(X_trainCalib)
    X_testCalib = preprocessor.transform(X_testCalib)
    return X_train, X_test, X_trainCalib, X_testCalib


def getCytoRNADataFromCsv(dataPath, batchesPath, batch1, batch2, trainPct = 0.8):
    data = genfromtxt(dataPath, delimiter=',', skip_header=0)
    batches = genfromtxt(batchesPath, delimiter=',', skip_header=0) 
    source = data[batches == batch1]
    target = data[batches == batch2]
    n_source = source.shape[0]
    p = np.random.permutation(n_source)
    cutPt = int(n_source * trainPct)
    source_train = source[p[:cutPt]]
    source_test = source[p[cutPt:]]
    n_target = target.shape[0]
    p = np.random.permutation(n_target)
    cutPt = int(n_target * trainPct)
    target_train = target[p[:cutPt]]
    target_test = target[p[cutPt:]]
    return source_train, source_test, target_train, target_test
    