#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:09:01 2019

@author: israelsolha
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = '/Users/israelsolha/Desktop/Data/Masters/Breast Cancer/breast-histopathology-images'
os.chdir(path)

def get_directory(image,df):
        id_ = df['ID'][df['File Name'] == image].values[0]
        target_ = str(df['Diagnostic'][df['File Name'] == image].values[0])
        return os.path.join(path,id_,target_,image)
        
def get_x(image,df):
        x = df['x'][df['File Name'] == image].values[0]
        return x-1
        
def get_y(image,df):
        y = df['y'][df['File Name'] == image].values[0]
        return y-1
        
        
def plot_img(patient = '9036', path = '/Users/israelsolha/Desktop/Data/Masters/Breast Cancer/breast-histopathology-images'):
    exclude = '.DS_Store'
    image_list = []
    path_1 = [i for i in os.listdir(os.path.join(path,patient)) if os.path.isdir(os.path.join(path,patient,i))]
    for path_ in path_1:
        image_list.extend([i for i in os.listdir(os.path.join(path,patient,path_)) if i!= exclude])
    
    id_ = [i.split('_')[0] for i in image_list]
    x = [int(i.split('_')[2].split('x')[1]) for i in image_list]
    y = [int(i.split('_')[3].split('y')[1]) for i in image_list]
    target = [int(i.split('_')[4].split('class')[1].split('.png')[0])for i in image_list]
    
    data_tuples = list(zip(image_list,id_,x,y,target))
    df = pd.DataFrame(data_tuples, columns = ['File Name','ID','x','y','Diagnostic'])
    df.sort_values(['x','y'],inplace = True)
    
    x_min = df.x.min()-1
    y_min = df.y.min()-1
    x_max = df.x.max()-1+50
    y_max = df.y.max()-1+50
    
    full_image = np.ones(shape=(y_max-y_min,x_max-x_min,3))
    for image in image_list:
       x_im = get_x(image,df)
       y_im = get_y(image,df)
       dir_im = get_directory(image,df)
       img = mpimg.imread(dir_im)
       full_image[y_im-y_min:y_im+img.shape[0]-y_min,x_im-x_min:x_im+img.shape[1]-x_min,0:3] = img
    plt.figure(dpi=2000)
    plt.imshow(full_image)

plot_img()