#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:09:10 2019

@author: ppxct2
Creating the training and testing data for the CNN CAS parameters 
"""

from sklearn.model_selection import train_test_split 
import numpy as np 
import matplotlib.pyplot as plt 

#Loading the data 
#Fits images for testing 
candels_data_all = np.load('/home/ppxct2/Documents/CANDELS/candels_data_all.npy')
#Normalised fits images 
normalised_data_all = np.load('/home/ppxct2/Documents/CANDELS/normalised_data_all.npy')
#Corresponding asymmetric values for training 
asym_data_all = np.load('/home/ppxct2/Documents/CANDELS/asym_data_all.npy')
#Corresponding concentration values for training 
conc_data_all = np.load('/home/ppxct2/Documents/CANDELS/conc_data_all.npy')
#Loading cropped images 
cropped_images = np.load('/home/ppxct2/Documents/CANDELS/cropped_images.npy')

### Plotting distrubtion of concentration values ###
plt.hist(conc_data_all, bins = 15, color = 'b', alpha = 0.75)
plt.xlabel('Concentration values')
plt.ylabel('Frequency')
plt.savefig('/home/ppxct2/Documents/CANDELS/concentration_dist.png')
plt.close()

#split data into training samples
#x_train = candels_data_all
#x_train = normalised_data_all
x_train = cropped_images 
#y_train = asym_data_all
y_train = conc_data_all

# Splitting training and validation 80% 20% - using random_state to ensure split the same way every time to avoid overfitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)
# Splitting testing and vaildation data 10% each 
x_test,x_val, y_test, y_val = train_test_split(x_val, y_val, test_size = 0.5, random_state = 42)

#creating array with rotated x_train images - rotated_CW
print(x_train.shape)
print(y_train.shape)
#Rotating all images clockwise 90 degrees 3 times 
rotated_CW = x_train.swapaxes(-2,-1)[...,::-1]
rotated_CW_2 = rotated_CW.swapaxes(-2,-1)[...,::-1]
rotated_CW_3 = rotated_CW_2.swapaxes(-2,-1)[...,::-1]

#saving the rotated images and all concentration/asym values in same order 
augmented_data = np.concatenate((x_train, rotated_CW, rotated_CW_2, rotated_CW_3), axis = 0)
augmented_vals = np.concatenate((y_train, y_train, y_train, y_train), axis = 0)
print('Adding rotated images array length is ', augmented_data.shape)
print('Corresponding values array length is' , augmented_vals.shape)

#saving all the data 
np.save('/home/ppxct2/Documents/CANDELS/x_train_data', x_train )
np.save('/home/ppxct2/Documents/CANDELS/y_train_data', y_train)
np.save('/home/ppxct2/Documents/CANDELS/x_test_data', x_test)
np.save('/home/ppxct2/Documents/CANDELS/y_test_data', y_test)
np.save('/home/ppxct2/Documents/CANDELS/x_val_data', x_val)
np.save('/home/ppxct2/Documents/CANDELS/y_val_data', y_val)
np.save('/home/ppxct2/Documents/CANDELS/augmented_data', augmented_data)
np.save('/home/ppxct2/Documents/CANDELS/augmented_vals', augmented_vals)