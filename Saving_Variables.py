#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:04:26 2019

@author: israelsolha
"""

from multiprocessing import Pool
import multiprocessing
import os
import pandas as pd
import numpy as np

path = '/root'
os.chdir(path)

print("Loading File Names")

exclude = '.DS_Store'
X_train_0 = [i for i in os.listdir(os.path.join(path,'train_dir','no_idc')) if i!=exclude]
X_train_0 = pd.DataFrame(X_train_0,columns=['File Name'])
X_train_0['Diagnostic'] = 0

X_train_1 = [i for i in os.listdir(os.path.join(path,'train_dir','has_idc')) if i!=exclude]
X_train_1 = pd.DataFrame(X_train_1,columns=['File Name'])
X_train_1['Diagnostic'] = 1

X_test_0 = [i for i in os.listdir(os.path.join(path,'test_dir','no_idc')) if i!=exclude]
X_test_0 = pd.DataFrame(X_test_0,columns=['File Name'])
X_test_0['Diagnostic'] = 0

X_test_1 = [i for i in os.listdir(os.path.join(path,'test_dir','has_idc')) if i!=exclude]
X_test_1 = pd.DataFrame(X_test_1,columns=['File Name'])
X_test_1['Diagnostic'] = 1

X_train = pd.concat([X_train_0, X_train_1], axis=0).reset_index(drop=True)
X_train['Directory'] = ["train_dir" for i in X_train['File Name']]
X_test = pd.concat([X_test_0, X_test_1], axis=0).reset_index(drop=True)


print("Creating distributed Dataset")

cases_1_train = len(X_train[X_train['Diagnostic']==1])
cases_1_test = len(X_test[X_test['Diagnostic']==1])

X_train = pd.DataFrame(np.concatenate([X_train[X_train['Diagnostic']==0].sample(cases_1_train),
                                       X_train[X_train['Diagnostic']==1]]),
                                       columns = ['File Name', 'Diagnostic','Directory']).sample(frac=1).reset_index(drop=True)

X_test = pd.DataFrame(np.concatenate([X_test[X_test['Diagnostic']==0].sample(cases_1_test),
                                      X_test[X_test['Diagnostic']==1]]),
                                      columns = ['File Name', 'Diagnostic']).sample(frac=1).reset_index(drop=True)


from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False,categories='auto')

y_train = np.array(X_train['Diagnostic'])
y_train = onehot_encoder.fit_transform(y_train.reshape(-1,1))

y_test = np.array(X_test['Diagnostic'])
y_test = onehot_encoder.fit_transform(y_test.reshape(-1,1))

train_path = os.path.join(path,'train_dir')
test_path = os.path.join(path,'test_dir')

print("Loading Images")

import cv2
import time

start = time.time()

def get_images(inp):
    if inp[0] == 'test':
        x = X_test
        path = test_path
    elif inp[0] == 'train':
        x = X_train
        path = train_path
    frac=inp[1]-1
    start=inp[2][0]
    end = inp[2][1]
    count = 0
    x = pd.DataFrame(np.array(x)[start:end+1])
    x_ret = np.empty((len(x), 50, 50,3), dtype=np.uint8)
    for i in range(len(x)):
        count+=1
        test = x.iloc[i,:]
        if test[1] == 0:
            full_path = os.path.join(path,'no_idc',test[0])
        elif test[1] == 1:
            full_path = os.path.join(path,'has_idc',test[0])
        img = cv2.imread(full_path,cv2.IMREAD_COLOR)
        if img.shape != (50,50):
            img = cv2.resize(img,(50,50))
        img = cv2.filter2D(img, -1, kernel_sharpening)
        if count%100==0:
            print(count,frac)
        x_ret[i, ...] = img
    return x_ret

n_cpu = multiprocessing.cpu_count()

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

def get_splits(x,n):
    a= np.array_split(range(len(np.array(x))),n)
    return [[i[0],i[-1]] for i in a]
    
a = get_splits(X_test,n_cpu)

p = Pool(n_cpu)

quer=(['test',fraction, a[fraction-1]] for fraction in range(1,n_cpu+1))
    
X_test = np.concatenate([i for i in p.map(get_images, quer)])
p.close()
p.join()
print("Loaded Test set")

p = Pool(n_cpu)

a = get_splits(X_train,n_cpu)
quer=(['train',fraction,a[fraction-1]] for fraction in range(1,n_cpu+1))

X_train = np.concatenate([i for i in p.map(get_images, quer)])
p.close()
p.join()

print("Loaded Training set")

print(f"Time to load files: {(time.time()-start):.1f}s")

os.chdir('..')

print()
print("Saving Variables")

import pickle

fileObject = open('/root/equal_variables_color_sharp','wb')
pickle.dump([X_train, X_test, y_train, y_test],fileObject)
fileObject.close() 

print()
print("Process complete")