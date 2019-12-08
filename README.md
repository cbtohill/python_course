# CB's project 

Clár-Bríd Tohill 

## Overview
MPAGS coursework.
Created Concolutional Neural Network to predict concentration values of Galaxies from fits images. 

## Files 
creates_data.py\
splitting_data.py\
candels_tensorflow.py\
pred_vs_true_plot.py\
cropped_images.npy\
conc_data_all.npy

## File overview 
**Create_data.py** - takes array of galaxy IDs and field position and iterates through image folders.\
Returns:\
three arrays of the same fits images, one with the unchanged images, 
one with the normalised images and one with the images cropped and normalised.\
**splitting_data.py** - splits fits images and corresponding concentration values into training, testing and validation data.\
Returns:\
Arrays of training, testing and validation data.\
**candels_tensorflow.py** - CNN to predict concentration values of galaxies from images.\ 
Returns:\
Predictions of concentration values of galaxies and true values.\
**pred_vs_true_plot.py** - plots the predictions of the concentration values against the true values in scatter plot and histogram. 

## Usage 
Use cropped_images.npy and conc_data_all.npy in splitting_data to create all of the training, testing and validation data. This can then be used to run the candels_tensorflow.py file which will output the data needed to plot the predictions vs True values in pred_vs_true_plot.py. 
