# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:34:46 2018

@author: Krzywy
"""

#This code will plot ROC curve for two different behaviours comparing distributions of their fluorescence 
#to estimate if neuron can discriminate between them. also calculates auROC.
import numpy as np
import pandas as pd
Final = pd.read_csv('Vmh4SF23beh1.csv', delimiter=';')
Beh = Final.loc[Final['Beh'] == 'Stretch']
nBeh = Final.loc[Final['Beh'] == 'Cornering']


import matplotlib.pyplot as plt

#plots histograms
fig, ax = plt.subplots(nrows=2, ncols=1)
Beh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='g', ax=ax[0])
nBeh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='b', ax=ax[0])

#plot cdf
Beh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.1, color='g', cumulative=True, normed=True, ax=ax[1])
nBeh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='b', cumulative=True, normed=True, ax=ax[1])

nBeh.loc[nBeh['Beh'] == 'Cornering', 'Beh'] = 0
Beh.loc[Beh['Beh'] == 'Stretch', 'Beh'] = 1
toconcat = [Beh, nBeh]
Fin = pd.concat(toconcat)
        
#Plotting ROC and calculating auROC
from sklearn import metrics
plt.figure(0).clf()

label = Final['Beh'].reset_index().values
label = label[:,1]
label = label.astype(int)
pred = Final['Mean(2)'].reset_index().values
pred = pred[:,1]
pred = pred.astype(float)
fpr, tpr, thresh = metrics.roc_curve(label, pred)
auc = metrics.roc_auc_score(label, pred)
plt.plot(fpr,tpr,label="Neuron, auc="+str(auc))

plt.legend(loc=0)

