# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 22:34:46 2018

@author: Krzywy
"""

#auROC 
import numpy as np
import pandas as pd
Final = pd.read_csv('Vmh4SF23beh1.csv', delimiter=';')
Beh = Final.loc[Final['Beh'] == 'Stretch']
nBeh = Final.loc[Final['Beh'] != 'Stretch']


import matplotlib.pyplot as plt

#plots histograms
fig, ax = plt.subplots(nrows=2, ncols=1)
Beh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='g', ax=ax[0])
nBeh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='b', ax=ax[0])

#plot cdf
Beh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.1, color='g', cumulative=True, normed=True, ax=ax[1])
nBeh['Mean(2)'].plot(kind='hist', bins=30, alpha=0.5, color='b', cumulative=True, normed=True, ax=ax[1])

Final.loc[Final['Beh'] != 'Stretch', 'Beh'] = 0
Final.loc[Final['Beh'] == 'Stretch', 'Beh'] = 1
        
#example code of plotting roc and calculatin auROC
#==============================================================================
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
#==============================================================================
# second method if i calculate stuff myself

#==============================================================================
# import matplotlib.pyplot as plt
# import numpy as np
# 
# x = # false_positive_rate
# y = # true_positive_rate 
# 
# # This is the ROC curve
# plt.plot(x,y)
# plt.show() 
# 
# # This is the AUC
# auc = np.trapz(y,x)
#==============================================================================
