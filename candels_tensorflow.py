#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:32:01 2019

@author: ppxct2
"""
#importing all modules needed 
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, AveragePooling2D
from tensorflow.keras.optimizers import RMSprop

# if multiple GPUs, only use one of them 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# avoid hogging all the GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

#loading the data for training and testing 
x_train = np.load('/home/ppxct2/Documents/CANDELS/x_train_data.npy')
y_train = np.load('/home/ppxct2/Documents/CANDELS/y_train_data.npy')
x_test = np.load('/home/ppxct2/Documents/CANDELS/x_test_data.npy')
y_test = np.load('/home/ppxct2/Documents/CANDELS/y_test_data.npy')
x_val = np.load('/home/ppxct2/Documents/CANDELS/x_val_data.npy')
y_val = np.load('/home/ppxct2/Documents/CANDELS/y_val_data.npy')
augmented_data = np.load('/home/ppxct2/Documents/CANDELS/augmented_data.npy')
augmented_vals = np.load('/home/ppxct2/Documents/CANDELS/augmented_vals.npy')

'''
#Data augmentation - if using rotated images uncomment this.
#using rotated images for training 
x_train = augmented_data
y_train = augmented_vals 
#shuffling data in unison so that the machine does not learn to predict the rotations 
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train, y_train)
'''
#taking a look at some data 
i = 11
plt.imshow(x_train[i], cmap = 'gray')
print('class = ', y_train[i])

print(x_train.shape[0], 'Training samples')
print(x_test.shape[0], 'test samples')
print(x_val.shape[0], 'validation samples')

#Modules needed 
def histplot(history):
    '''
    Function to plot the loss(RMSE) and the mean absolute error (MAE) of the 
    training and testing data against the number of epochs. 
    Also plots the minimum RMSE and maximum MAE of the test data.  
    '''
    hist = pd.DataFrame(history.history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    hist.plot(y=['loss', 'val_loss'], ax=ax1)
    min_loss = hist['val_loss'].min()
    ax1.hlines(min_loss, 0, len(hist), linestyle='dotted',
               label='min(val_loss) = {:.3f}'.format(min_loss))
    ax1.legend(loc='upper right')
    hist.plot(y=['mean_absolute_error', 'val_mean_absolute_error'], ax=ax2)
    max_acc = hist['val_mean_absolute_error'].max()
    ax2.hlines(max_acc, 0, len(hist), linestyle='dotted',
               label='max(val_mean_absolute_error) = {:.3f}'.format(max_acc))
    ax2.legend(loc='upper right')

def root_mean_squared_error(y_true, y_pred):
    '''
    Function to calculate the root mean squared error of the predictions
    from the network. 
    
    '''
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

# better-looking plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams['font.size'] = 14
    
# Reshaping data 
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_val = np.expand_dims(x_val, axis = 3)

# a fairly small network for speed
num_classes = 1 # only want one value out - concentration
### CNN ### 
cnnmodel = Sequential()
cnnmodel.add(Conv2D(4, (3, 3), activation='relu', input_shape=(31, 31, 1)))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(Conv2D(8, (3, 3), activation='relu'))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
cnnmodel.add(AveragePooling2D((2, 2)))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(16, activation = 'relu'))
cnnmodel.add(Dense(num_classes, activation='linear'))

#define batch size and number of epochs - compile the model
batch_size = 128
epochs = 30
cnnmodel.compile(loss= root_mean_squared_error ,optimizer=RMSprop(), metrics=['mean_absolute_error'])
cnnmodel.summary()
# save weights for reinitialising on asymmetry network
cnnmodel.save_weights('/tmp/cnnmodel_init_weights.tf')

history = cnnmodel.fit(x_train, y_train,batch_size=batch_size, epochs=3*epochs,verbose=2,validation_data=(x_test, y_test))
histplot(history)
plt.savefig('/home/ppxct2/Documents/CANDELS/MAE_RMSE_CONC.png')

#prediction of concentration parameter - saving to plot against true values
pred = cnnmodel.predict(x_val)
true = y_val
np.save('/home/ppxct2/Documents/CANDELS/predict', pred)
np.save('/home/ppxct2/Documents/CANDELS/true', true)

#plotting score of network where Test Loss is the RMSE
score = cnnmodel.evaluate(x_val, y_val, verbose=2)
print('Test loss:', score[0])
print('Test Mean absolute error:', score[1])