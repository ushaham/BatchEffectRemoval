##================##
##  Calibration   ##
## Drop-seq Data2 ##
##    Jun Zhao    ##
##================##

## This script is for the calibration analysis on single cell RNAseq data
## Data used is from Shekhar et al. (2016)
## Code includes preprocessing, calibration with ComBat (and some others), plotting and comparisons


setwd("/raid3/DropSeq/Retina/")
source("./Second/funs.R")

rawData2 <- read.table("./GSE81904_BipolarUMICounts_Cell2016.txt.gz",
                       header = T, stringsAsFactors = F, sep = "\t", row.names = 1)
mean(colMeans(rawData2 == 0)) #~97%

filteredData2 <- rawData2[,colSums(rawData2 > 0) > 500]
mt_genes <- grep("mt-", rownames(filteredData2), value = TRUE)
cells_use <- colnames(filteredData2)[colSums(filteredData2[mt_genes, ])/colSums(filteredData2) < 0.1]
filteredData2 <- filteredData2[, cells_use]

filteredData2 <- filteredData2[rowSums(filteredData2 > 0) > 30 & rowSums(filteredData2) > 60 ,]

normData2 <- log(sweep(filteredData2, 2, colSums(filteredData2), "/")*1e4 + 1)
write.table(normData2, file = "./normData2.csv", quote = F, sep = ",", row.names = F, col.names = F)

batch2 <- gsub("_[ATCGN]{12}", "", colnames(filteredData2))
batch2[batch2 == "Bipolar5" | batch2 == "Bipolar6"] = 2
batch2[batch2 != 2] = 1
cat(batch2, file = "./Second/1215/batch.csv", sep = "\n")

# calculate correlation between batches
batchCorData2 <- CalBatchCor(normData2, batch2)
CorHeat(batchCorData2)

# ComBat on normalized data
combatData2 <- ComBat(normData2, batch = batch2, prior.plots = F, par.prior = T)
batchCorData2_after <- CalBatchCor(combatData2, batch2)
CorHeat(batchCorData2_after)

# Center the rows
rmData2 <- rowMeans(normData2)
scaleData2 <- sweep(normData2, 1, rmData2)

# PCA on data
pcaData2 <- fastPCA(t(scaleData2),k = 50)
plot(diag(pcaData2$S)/pcaData2$S[1])
pcData2 <- pcaData2$U[,1:37] %*% pcaData2$S[1:37,1:37]

# plot batch on PC space
ggplot() + geom_point(data = data.frame(x=pcData2[,1], y=pcData2[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = plot_col,name="Group") + labs(x="Dim1", y="Dim2") + theme_bw()
ScatterHist(pcData2, batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(pcData2[,5])-1, max(pcData2[,5])+1), ylim = c(min(pcData2[,7])-1, max(pcData2[,7])+1), 1)
apply(pcData2, 2, cor, as.numeric(batch2))

# plot after ComBat on PC space
rmCData2 <- rowMeans(combatData2)
scaleCData2 <- sweep(combatData2, 1, rmCData2)
pcCData2 <- as.matrix(t(scaleCData2)) %*% pcaData2$V[,1:37]
ScatterHist(pcCData2, batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(pcCData2[,5])-1, max(pcCData2[,5])+1), ylim = c(min(pcCData2[,7])-1, max(pcCData2[,7])+1), 1)
apply(pcCData2, 2, cor, as.numeric(batch2))

# 37 PCs for t-SNE
set.seed(11111)
tsneData2 <- Rtsne(pcData2, dims = 2)
apply(tsneData2$Y, MARGIN = 2, cor, as.numeric(batch2))
# plot batch on tSNE map
ggplot() + geom_point(data = data.frame(x=tsneData2$Y[,1], y=tsneData2$Y[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = plot_col,name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()

# tSNE after ComBat
set.seed(11111)
tsneCData2 <- Rtsne(pcCData2, dims = 2)
ggplot() + geom_point(data = data.frame(x=tsneCData2$Y[,1], y=tsneCData2$Y[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = plot_col,name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()

# check biological markers
OvrLyExp(data2D = data.frame(x=tsneCSData2$Y[,1], y=tsneCSData2$Y[,2]), dataexp = combatData2, gene = "Rho")
OvrLyExp(data2D = data.frame(x=tsneCSData2$Y[,1], y=tsneCSData2$Y[,2]), dataexp = lrkData2, gene = "Apoe")

write.table(scale(pcData2), file = "./Second/1215/Data2_standardized_37PCs.csv", 
            quote = F, sep = ",", row.names = F, col.names = F)
write.table(scale(pcCData2), file = "./Second/Data2_CombatStandardized_37PCs.csv", 
            quote = F, sep = ",", row.names = F, col.names = F)


##================##
##  MMD on Data2  ##
##================##

# Calibrated by MMD
pcMMDData2 <- read.table("./Second/1215/RNA_second_calibratedData1215.csv", header = F, sep = ",", stringsAsFactors = F)
cat(apply(scale(pcData2), 2, cor, as.numeric(batch2)))
cat(apply(scale(pcCData2), 2, cor, as.numeric(batch2)))
cat(apply(pcMMDData2, 2, cor, as.numeric(batch2)))

# check target
pcMMDData2_target <- pcMMDData2[batch2 == 1,]
mean(pcMMDData2_target == pcaData2$U[,1:37])

# compare with before and ComBat
ScatterHist(scale(pcData2), batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(scale(pcData2)[,5])-1, max(scale(pcData2)[,5])+1), 
            ylim = c(min(scale(pcData2)[,7])-1, max(scale(pcData2)[,7])+1), .1)
ScatterHist(scale(pcCData2), batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(scale(pcCData2)[,5])-1, max(scale(pcCData2)[,5])+1), 
            ylim = c(min(scale(pcCData2)[,7])-1, max(scale(pcCData2)[,7])+1), .1)
ScatterHist(scale(pcMMDData2), batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(scale(pcMMDData2)[,5])-1, max(scale(pcMMDData2)[,5])+1), 
            ylim = c(min(scale(pcMMDData2)[,7])-1, max(scale(pcMMDData2)[,7])+1), .1)

ScatterHist(scale(pcData2), batch2, 1, 2, 1, 2, "PC1", "PC2", 
            xlim = c(min(scale(pcData2)[,1])-1, max(scale(pcData2)[,1])+1), 
            ylim = c(min(scale(pcData2)[,2])-1, max(scale(pcData2)[,2])+1), .1)
ScatterHist(scale(pcCData2), batch2, 1, 2, 1, 2, "PC1", "PC2", 
            xlim = c(min(scale(pcCData2)[,1])-1, max(scale(pcCData2)[,1])+1), 
            ylim = c(min(scale(pcCData2)[,2])-1, max(scale(pcCData2)[,2])+1), .1)
ScatterHist(scale(pcMMDData2), batch2, 1, 2, 1, 2, "PC1", "PC2", 
            xlim = c(min(scale(pcMMDData2)[,1])-1, max(scale(pcMMDData2)[,1])+1), 
            ylim = c(min(scale(pcMMDData2)[,2])-1, max(scale(pcMMDData2)[,2])+1), .1)

# t-SNE for MMD pcData2
set.seed(11111)
tsneMMDData2 <- Rtsne(pcMMDData2, dims = 2)
ggplot() + geom_point(data = data.frame(x=tsneMMDData2$Y[,1], y=tsneMMDData2$Y[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
apply(tsneMMDData2$Y, 2, cor, as.numeric(batch2))
ScatterHist(tsneMMDData2$Y, batch2, 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneMMDData2$Y[,1])-1, max(tsneMMDData2$Y[,1])+1), 
            ylim = c(min(tsneMMDData2$Y[,2])-1, max(tsneMMDData2$Y[,2])+1), 1)

tsneSData2 <- Rtsne(scale(pcData2), dims = 2)
ggplot() + geom_point(data = data.frame(x=tsneSData2$Y[,1], y=tsneSData2$Y[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
apply(tsneSData2$Y, 2, cor, as.numeric(batch2))
ScatterHist(tsneSData2$Y, batch2, 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneSData2$Y[,1])-1, max(tsneSData2$Y[,1])+1), 
            ylim = c(min(tsneSData2$Y[,2])-1, max(tsneSData2$Y[,2])+1), 1)

tsneCSData2 <- Rtsne(scale(pcCData2), dims = 2)
ggplot() + geom_point(data = data.frame(x=tsneCSData2$Y[,1], y=tsneCSData2$Y[,2], group=as.factor(batch2)), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
apply(tsneCSData2$Y, 2, cor, as.numeric(batch2))
ScatterHist(tsneCSData2$Y, batch2, 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneCSData2$Y[,1])-1, max(tsneCSData2$Y[,1])+1), 
            ylim = c(min(tsneCSData2$Y[,2])-1, max(tsneCSData2$Y[,2])+1), 1)

# compare each pair of replicate..
label2 <- gsub("_[ATCGN]{12}", "", colnames(filteredData2))
label2 <- as.numeric(gsub("Bipolar", "", label2))
table(label2)
pdf("./Second/CompareReps.pdf")
for(i in 1:5){
  for(j in (i+1):6){
    cells_use <- which(label2 == i | label2 == j)
    ScatterHist(tsneSData2$Y[cells_use,], label2[cells_use], i, j, 1, 2, "Dim1", "Dim2", 
                xlim = c(min(tsneSData2$Y[cells_use,1])-1, max(tsneSData2$Y[cells_use,1])+1), 
                ylim = c(min(tsneSData2$Y[cells_use,2])-1, max(tsneSData2$Y[cells_use,2])+1), 1)
    ScatterHist(tsneCSData2$Y[cells_use,], label2[cells_use], i, j, 1, 2, "Dim1", "Dim2", 
                xlim = c(min(tsneCSData2$Y[cells_use,1])-1, max(tsneCSData2$Y[cells_use,1])+1), 
                ylim = c(min(tsneCSData2$Y[cells_use,2])-1, max(tsneCSData2$Y[cells_use,2])+1), 1)
    ScatterHist(tsneMMDData2$Y[cells_use,], label2[cells_use], i, j, 1, 2, "Dim1", "Dim2", 
                xlim = c(min(tsneMMDData2$Y[cells_use,1])-1, max(tsneMMDData2$Y[cells_use,1])+1), 
                ylim = c(min(tsneMMDData2$Y[cells_use,2])-1, max(tsneMMDData2$Y[cells_use,2])+1), 1)
  }
}
dev.off()

# biological validation for MMD, plot markers
# destandardize PC
cmPcData2 <- colMeans(pcData2)
sigPcData2 <- apply(pcData2, MARGIN = 2, sd)
pcMMDDData2 <- sweep(pcMMDData2, MARGIN = 2, STATS = sigPcData2, FUN = "*")
pcMMDDData2 <- sweep(pcMMDDData2, MARGIN = 2, STATS = cmPcData2, FUN = "+")
caliData2 <- as.matrix(pcMMDDData2) %*% t(pcaData2$V[,1:37])
caliData2 <- t(caliData2)
rownames(caliData2) <- rownames(normData2)
lrkData2 <- t(pcaData2$U[,1:37] %*% pcaData2$S[1:37,1:37] %*% t(pcaData2$V[,1:37]))
rownames(lrkData2) <- rownames(normData2)
OvrLyExp(data2D = data.frame(x=tsneMMDData2$Y[,1], y=tsneMMDData2$Y[,2]), dataexp = caliData2, gene = "Apoe")
OvrLyExp(data2D = data.frame(x=tsneMMDData2$Y[,1], y=tsneMMDData2$Y[,2]), dataexp = normData2, gene = "Prkca")

# Compare only cells with high expression in Apoe and Prkca
cells_Apoe <- which(normData2["Apoe", ] > 4.5)
cells_Apoe <- which(normData2["Prkca", ] > 3)
# before
ggplot() + geom_point(data = data.frame(x=tsneSData2$Y[cells_Apoe,1], y=tsneSData2$Y[cells_Apoe,2], 
                                        group=as.factor(batch2[cells_Apoe])), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
ScatterHist(tsneSData2$Y[cells_Apoe,], batch2[cells_Apoe], 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneSData2$Y[,1])-1, max(tsneSData2$Y[,1])+1), 
            ylim = c(min(tsneSData2$Y[,2])-1, max(tsneSData2$Y[,2])+1), 1)
# ComBat
ggplot() + geom_point(data = data.frame(x=tsneCSData2$Y[cells_Apoe,1], y=tsneCSData2$Y[cells_Apoe,2], 
                                        group=as.factor(batch2[cells_Apoe])), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
ScatterHist(tsneCSData2$Y[cells_Apoe,], batch2[cells_Apoe], 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneCSData2$Y[,1])-1, max(tsneCSData2$Y[,1])+1), 
            ylim = c(min(tsneCSData2$Y[,2])-1, max(tsneCSData2$Y[,2])+1), 1)
# MMD
ggplot() + geom_point(data = data.frame(x=tsneMMDData2$Y[cells_Apoe,1], y=tsneMMDData2$Y[cells_Apoe,2], 
                                        group=as.factor(batch2[cells_Apoe])), 
                      aes(x=x,y=y, col=as.factor(group)),size=0.5, alpha=0.75) + 
  scale_colour_manual(values = c("red","blue"),name="Batch") + labs(x="Dim1", y="Dim2") + theme_bw()
ScatterHist(tsneMMDData2$Y[cells_Apoe,], batch2[cells_Apoe], 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneMMDData2$Y[,1])-1, max(tsneMMDData2$Y[,1])+1), 
            ylim = c(min(tsneMMDData2$Y[,2])-1, max(tsneMMDData2$Y[,2])+1), 1)


## Remove batch effect by z-scoring
batch1Data2 <- normData2[,batch2 == 1]
batch2Data2 <- normData2[,batch2 == 2]
rmB1Data2 <- rowMeans(batch1Data2)
rmB2Data2 <- rowMeans(batch2Data2)
sigB1Data2 <- apply(batch1Data2, MARGIN = 1, FUN = sd)
sigB2Data2 <- apply(batch2Data2, MARGIN = 1, FUN = sd)
sigB2Data2[sigB2Data2 == 0] <- 1
stdB2Data2 <- sweep(batch2Data2, MARGIN = 1, rmB2Data2, FUN = "-")
stdB2Data2 <- sweep(stdB2Data2, MARGIN = 1, sigB2Data2, FUN = "/")
zsB2Data2 <- sweep(stdB2Data2, 1, sigB1Data2, FUN = "*")
zsB2Data2 <- sweep(zsB2Data2, 1, rmB1Data2, FUN = "+")
zsData2 <- cbind(batch1Data2, zsB2Data2)
rmZData2 <- rowMeans(zsData2)
scaleZData2 <- sweep(zsData2, 1, rmZData2)
pcZData2 <- as.matrix(t(scaleZData2)) %*% pcaData2$V[,1:37]
ScatterHist(scale(pcZData2), batch2, 1, 2, 1, 2, "PC1", "PC2", 
            xlim = c(min(scale(pcZData2)[,1])-1, max(scale(pcZData2)[,1])+1), 
            ylim = c(min(scale(pcZData2)[,2])-1, max(scale(pcZData2)[,2])+1), .1)
ScatterHist(scale(pcZData2), batch2, 1, 2, 5, 7, "PC5", "PC7", 
            xlim = c(min(scale(pcZData2)[,5])-1, max(scale(pcZData2)[,5])+1), 
            ylim = c(min(scale(pcZData2)[,7])-1, max(scale(pcZData2)[,7])+1), .1)
apply(scale(pcZData2), 2, cor, as.numeric(batch2))
write.table(scale(pcZData2), file = "./Second/Data2_ZscoreStandardized_37PCs.csv", 
            quote = F, sep = ",", row.names = F, col.names = F)
# t-SNE
set.seed(11111)
tsneZData2 <- Rtsne(scale(pcZData2), dims = 2)
apply(tsneZData2$Y, 2, cor, as.numeric(batch2))
ScatterHist(tsneZData2$Y, batch2, 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneZData2$Y[,1])-1, max(tsneZData2$Y[,1])+1), 
            ylim = c(min(tsneZData2$Y[,2])-1, max(tsneZData2$Y[,2])+1), 1)


# Remove highly correlated PCs
pcNData2 <- pcData2[, abs(apply(pcData2, 2, cor, as.numeric(batch2))) < 0.2]
write.table(scale(pcNData2), file = "./Second/Data2_RmPCsStandardized_34PCs.csv", 
            quote = F, sep = ",", row.names = F, col.names = F)
set.seed(11111)
tsneNData2 <- Rtsne(scale(pcNData2), dims = 2)
apply(tsneNData2$Y, 2, cor, as.numeric(batch2))
ScatterHist(tsneNData2$Y, batch2, 1, 2, 1, 2, "Dim1", "Dim2", 
            xlim = c(min(tsneNData2$Y[,1])-1, max(tsneNData2$Y[,1])+1), 
            ylim = c(min(tsneNData2$Y[,2])-1, max(tsneNData2$Y[,2])+1), 1)