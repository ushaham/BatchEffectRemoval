# Calibration

script for the paper "Removal of Batch Effects using Distribution-Matching ResidualNetworks" by Uri Shaham, Kelly P. Stanton, Jun Zhao, Huamin Li, Ruth Montgomery,and Yuval Kluger

The script Calibration_demo loads to CyTOF datasets, corresponding to measurements of blood of the same person in two different CyTOF machine. 
The script trains a MMD-ResNet using one of the datasets as source and the other as target, to remove the batch effects.