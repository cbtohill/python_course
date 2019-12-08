#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:27:51 2019

@author: ppxct2
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table

### Loading the data #####
goods_ID_all = np.load('/home/ppxct2/Documents/CANDELS/goods_ID_all.npy')
#Creating paths to two folders of images as IDs overlap so cannot be in one folder 
path_north = '/home/ppxct2/Documents/CANDELS/GOODSN_opt' #Path to GOODSN image folder
path_south = '/home/ppxct2/Documents/CANDELS/GOODSS_opt' #Path to GOODSS image folder 
goods_table = Table.read('/home/ppxct2/Documents/CANDELS/goods_table.fits')

#Creating array of all of the images 
def image_array(goods_ID_all):
    '''
    Function to iterate through a 2D array of IDs and field names to match
    image to ID and append to an array.
    Parameters:
    ----------------------------
    goods_ID_all: array like 
    a = ID : float 
        This is the ID of each galaxy, these IDs overlap between GOODSN and 
        GOODSS so must be separted based on field first. 
    b = field : string 
        This is the field where each galaxy is located
        
    path_north: path to folder containing all galaxy images from GOODSN field
    path_south: path to folder containing all galaxy images from GOODSS field
    
    Returns:
    ------------------------------
    images_all:
        array contatining each fits image as a 2D array in order that ID was inputted
    
    '''
    images_all = []
    for a, b in goods_ID_all:
        a = int(a) 
        if b == 'gn': #GOODSN images 
            image_file = fits.open(path_north+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()
            images_all.append(image_data)
            del image_file
        elif b == 'gs': #GOODSS images
            image_file = fits.open(path_south+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()    
            images_all.append(image_data)
            del image_file
    return(np.array(images_all))
    
            
#Creating array of all of the images normalised   
def images_normalised(goods_ID_all): 
    '''
    Function to iterate through a 2D array of IDs and field names to match
    image to ID, normalise image and append to an array.
    Parameters:
    ----------------------------
    goods_ID_all: array like 
    a = ID : float 
        This is the ID of each galaxy, these IDs overlap between GOODSN and 
        GOODSS so must be separted based on field first. 
    b = field : string 
        This is the field where each galaxy is located
        
    path_north: path to folder containing all galaxy images from GOODSN field
    path_south: path to folder containing all galaxy images from GOODSS field
    
    Returns:
    -----------------------------
    images_all_norm:
        array contatining each fits image normalised so maximum pixel value is 1
        as a 2D array in order that ID was inputted
    
    '''
    images_all_norm = []
    for a, b in goods_ID_all:
        a = int(a)
        if b == 'gn': #GOODSN images 
            image_file = fits.open(path_north+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()
            image_data_norm = image_data/np.max(image_data)
            images_all_norm.append(image_data_norm)
            del image_file
        elif b == 'gs': #GOODSS images 
            image_file = fits.open(path_south+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()    
            image_data_norm = image_data/np.max(image_data)
            images_all_norm.append(image_data_norm)
            del image_file
    return(np.array(images_all_norm))


#Creating array of images cropped and normalised 
def cropped_images(goods_ID_all):
    '''
    Function to iterate through a 2D array of IDs and field names to match
    image to ID then crop images from 101 x 101 pixels to 31 x 31 pixels,
    normalise and append to an array.
    Parameters:
    ----------------------------
    goods_ID_all: array like 
    a = ID : float 
        This is the ID of each galaxy, these IDs overlap between GOODSN and 
        GOODSS so must be separted based on field first. 
    b = field : string 
        This is the field where each galaxy is located
        
    path_north: path to folder containing all galaxy images from GOODSN field
    path_south: path to folder containing all galaxy images from GOODSS field
    
    Returns:
    ------------------------------
    images_all_crop:    
        array contatining each fits image cropped to 31 x 31 pixels and normlaised
        as a 2D array in order that ID was inputted
    
    '''
    images_all_crop = []
    for a, b in goods_ID_all:
        a = int(a)
        if b == 'gn': #GOODSN images 
            image_file = fits.open(path_north+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()
            image_crop = image_data[35:66,35:66]
            norm_crop = image_crop/(np.max(image_crop))
            images_all_crop.append(norm_crop)
            del image_file
        elif b == 'gs': #GOODSS images 
            image_file = fits.open(path_south+'/img_'+str(a)+'.fits')
            image_data = image_file[0].data.copy()  
            image_crop = image_data[35:66,35:66]
            norm_crop = image_crop/(np.max(image_crop))
            images_all_crop.append(norm_crop)
            del image_file
    return(np.array(images_all_crop))

  
images_all = image_array(goods_ID_all)
np.save('/home/ppxct2/Documents/CANDELS/candels_data_all', images_all)

images_all_norm = images_normalised(goods_ID_all)
np.save('/home/ppxct2/Documents/CANDELS/normalised_data_all', images_all_norm)

images_all_crop = cropped_images(goods_ID_all)        
np.save('/home/ppxct2/Documents/CANDELS/cropped_images', images_all_crop)  

