#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:12:29 2019

@author: ppxct2
"""
import numpy as np
import matplotlib.pyplot as plt
# better-looking plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['font.size'] = 14

#loading predicted and true values outputted from network
x = np.load('/home/ppxct2/Documents/CANDELS/predict.npy')
y = np.load('/home/ppxct2/Documents/CANDELS/true.npy')
plt.plot(x, y, linestyle = 'none', marker = 'x', color = 'k', alpha = 1)
'''
#### Asymmetry #####
plt.plot((-0.1,0.6), (-0.1,0.6))
plt.xlim(-0.1, 0.6)
plt.ylim(-0.1, 0.6)
plt.xlabel('Predicted Asymmetry Values')
plt.ylabel('True Asymmetry values')
plt.savefig('/home/ppxct2/Documents/CANDELS/pred_vs_true_A.png')
'''
### Concentration #### 
plt.plot((0.0, 5), (0.0,5))
plt.xlabel('Predicted Concentration Values')
plt.ylabel('True Concentration values')
plt.xlim(0, max(y)+1)
plt.ylim(0, max(y)+1)
#plt.show()
plt.savefig('/home/ppxct2/Documents/CANDELS/pred_vs_true_C.png')
plt.close()

#Plotting histogram of predicted values against true values 
#Creating bin widths so that predicitons and true values can be compared
xbins = np.arange(0,4.5,0.25)
#printing any predictions outside range of true values
for i in x:
    if i < np.min(xbins) or i > np.max(xbins):
        print('Prediction of concentration outside range plotted:', i)

plt.hist(x, color = 'g', alpha = 0.75, label = 'Predicted Values', bins = (xbins))
plt.xlabel('Concentration Values')
plt.ylabel('Frequency')
plt.hist(y, color= 'k', alpha = 0.7, label = 'True Values', bins=(xbins), histtype = 'step', linewidth = 3)
plt.legend()
plt.savefig('/home/ppxct2/Documents/CANDELS/hist_predictions.png')