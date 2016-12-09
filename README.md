# Calibration

Script for the paper "Removal of Batch Effects using Distribution-Matching ResidualNetworks" by Uri Shaham, Kelly P. Stanton, Jun Zhao, Huamin Li, Ruth Montgomery, and Yuval Kluger

The script CyTOF_calibration.py loads two CyTOF datasets, corresponding to measurements of blood of the same person on the same machine in two different days. The script trains a MMD-ResNet using one of the datasets as source and the other as target, to remove the batch effects. once the net is trained, the results are evaluated. It produces the results described in Section 4.2.1.

The script Comparison_linear.py compares the performance of MMD-ResNet on CyTOF data to (1) calibration by shifting and rescaling each marker and (2) removing the principal component most correlated with the batch. It produces the results described in Appendix B

The script checkGeneralization.py loads CyTOF data of two patients on the same machine, on two different days (each patient was measured on both days). We train a MMD-ResNet for each patient and then check how well it calibrates the data of the other patient. It produces the results described in Section 4.2.2. 

The script CyTOF_calibration_noSkip_comparison has a comparison between MMD-ResNet and standard MLP on CyTOF data

The scripts of the RNA experiment are omitted from this github repository due to the size of the datasets. 
The data and scripts can be shared upon request. 

All scripts are written in Keras.

Any questions should be refered to Uri Shaham, uri.shaham@yale.edu.