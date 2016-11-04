from sklearn import decomposition
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras import backend as K
import numpy as np

#object to monitor progress of a deep net and print other relevant info
class monitor(Callback):
    #keras calls this when training begins
    def on_train_begin(self, logs={}):
        Callback.on_train_begin(self, logs=logs)
        #initialize list of losses
        self.losses = []
        self.val_losses = []
        #initialize plotting
        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
    #Keras calls this at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        Callback.on_epoch_end(self, epoch, logs=logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        #plot the cost for training and testing so far
        lossHandle, = self.ax.plot(self.losses,color='blue', label = 'loss')
        val_lossHandle, = self.ax.plot(self.val_losses,color='red', label = 'validation loss')
        self.ax.legend(handles = [lossHandle, val_lossHandle])
        plt.draw()
        plt.pause(0.01)
        
#TODO this currently uses the last layer of the net which isnt what you want if you are doing MMD on
#a middle layer        
class monitorMMD(monitor):
    def __init__(self,inputData,MMDtarget,netMMDLayerPredict=None):
        self.MMDtarget = MMDtarget
        self.inputData = inputData
        self.netMMDLayerPredict = netMMDLayerPredict
        if self.netMMDLayerPredict==None:
            self.netMMDLayerPredict = self.model.predict
    def on_train_begin(self, logs={}):
        monitor.on_train_begin(self, logs=logs)
        
        fig3 = plt.figure()
        self.axFig3 = fig3.add_subplot(111)
        
        #DO PCA
        self.pca = decomposition.PCA(n_components=2)
        self.pca.fit(self.MMDtarget)
        self.MMDtargetEmbedding = np.dot(self.MMDtarget,self.pca.components_[[0,1]].transpose())
        
    def on_epoch_end(self, epoch, logs={}):
        monitor.on_epoch_end(self, epoch, logs=logs)
        #clear plot
        self.axFig3.cla()
        #plot MMD target embbeding
        MMDtargetEmbeddingHandle = self.axFig3.scatter(self.MMDtargetEmbedding[:,0], self.MMDtargetEmbedding[:,1],
                                                        alpha=0.25, s=10, cmap='rainbow',
                                                        label="MMD target embedding")
        
        #plot network output projected on target embedding
        plotPredictions = self.netMMDLayerPredict(self.inputData)
        projection = np.dot(plotPredictions,self.pca.components_[[0,1]].transpose())
        NetOuputHandle = self.axFig3.scatter(projection[:,0],projection[:,1], color='red', alpha=0.25, s=10,
                                              label='Net output projected on target embedding')
        self.axFig3.legend(handles = (MMDtargetEmbeddingHandle, NetOuputHandle))
        plt.draw()
        plt.pause(0.01)
        
class monitorAnchor(Callback):
    def __init__(self,xInput, yInput, xTarget,yTarget,netAnchorLayerPredict):
        self.xInput = xInput
        self.yInput = yInput
        self.xTarget = xTarget
        self.yTarget = yTarget
        self.netAnchorLayerPredict = netAnchorLayerPredict
        if self.netAnchorLayerPredict==None:
            self.netAnchorLayerPredict = self.model.predict
            
    def on_train_begin(self, logs={}):
        Callback.on_train_begin(self, logs=logs)
        
        fig = plt.figure()
        self.axFig = fig.add_subplot(111)
        
        #DO PCA
        self.pca = decomposition.PCA(n_components=2)
        self.pca.fit(self.xTarget)
        self.targetEmbedding = np.dot(self.xTarget,self.pca.components_[[0,1]].transpose())
        
    def on_epoch_end(self, epoch, logs={}):
        Callback.on_epoch_end(self, epoch, logs=logs)
        #clear plot
        self.axFig.cla()
        #plot target embbeding
        targetEmbeddingHandle = self.axFig.scatter(self.targetEmbedding[:,0], self.targetEmbedding[:,1],
                                                        alpha=0.25, s=10, c=self.yTarget, cmap="rainbow",
                                                        label="MMD target embedding")
        
        #plot network output projected on target embedding
        plotPredictions = self.netAnchorLayerPredict(self.xInput)
        #print(plotPredictions)
        projection = np.dot(plotPredictions,self.pca.components_[[0,1]].transpose())
        NetOuputHandle = self.axFig.scatter(projection[:,0],projection[:,1], 
                                            c=self.yInput, cmap="rainbow",
                                            alpha=0.25, s=10,
                                              label='Net output projected on target embedding')
        self.axFig.legend(handles = (targetEmbeddingHandle, NetOuputHandle))
        plt.draw()
        plt.pause(0.01)
    