#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:17:28 2019

@author: israelsolha
"""

import pickle
import numpy as np


file_Name = 'equal_variables_color_sharp'
fileObject = open(file_Name,'rb')  
X_train, X_test, y_train, y_test = pickle.load(fileObject)

# X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[3],X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[2],X_test.shape[3],X_test.shape[1])

print("Files loaded")

X_train = X_train/255
X_test = X_test/255

print("Normalization ended")

# # def get_mean(i=3):
# #     for i in range(3):
# #         print("Removing mean from channel %d" %i)
# #         yield (np.sum(X_train[:,:,:,i]) + 
# #             np.sum(X_test[:,:,:,i]))/(
# #             (len(X_test) + len(X_train)*2500)),i
    
# # for mean,i, in get_mean():
# #     X_train[:,:,:,i] = X_train[:,:,:,i] - mean
# #     X_test[:,:,:,i] = X_test[:,:,:,i] - mean  


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, CSVLogger
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 50
batch_size = 64

classifier = Sequential()

classifier.add(Conv2D(32,kernel_size = 3, input_shape = (IMAGE_SIZE,IMAGE_SIZE,3),padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(32,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(32,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(64,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(64,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.3))

classifier.add(Conv2D(128,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(128,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(Conv2D(128,kernel_size = 3, padding="same",
                      activation='relu',
                      kernel_regularizer=l2(0.0),
                      kernel_initializer='he_normal'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=2))
classifier.add(Dropout(0.3))

classifier.add(Flatten())
classifier.add(Dense(256, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))
classifier.add(Dense(2, activation = 'sigmoid'))

epochs = 60
model_name = 'model.hdf5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(model_name, save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5, mode='min')
csv_logger = CSVLogger('training_ap4.log')
optimizer = Adam(learning_rate=0.001)

classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train,y_train, batch_size = batch_size, 
                callbacks=[earlyStopping,mcp_save,reduce_lr_loss,csv_logger],
                validation_split = 0.2,
                epochs = epochs)

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

model = load_model('model.hdf5')

y_pred = model.predict(X_test,verbose=1)

from sklearn.metrics import accuracy_score
print()
print()
y_max = [max(i) for i in y_pred]
for conf in [0.8, 0.95, 0.99, 0.999, 0.9999]:
    y_prede = []
    y_teste = []
    for ind,val in enumerate(y_max):
        if val >= conf:
            y_prede.append(y_pred[ind].tolist())
            y_teste.append(y_test[ind].tolist())
    cm = confusion_matrix(np.array(y_teste).argmax(axis=1),np.array(y_prede).argmax(axis=1))
    
    print(f"Confidence: " + f"{conf*100}".rstrip('0').rstrip('.') + "%")
    print(f"Covered cases: {(100*len(y_teste)/len(y_test)):.2f}%")
    print()
    print(cm)
    print()
    if len(cm)>1:
        if cm[1,0]+cm[1,1] != 0:
            print(f"Recall: {(100*(cm[1,1])/(cm[1,0]+cm[1,1])):.2f}%")
        if cm[0,1]+cm[1,1] != 0:
            print(f"Precision: {(100*(cm[1,1])/(cm[0,1]+cm[1,1])):.2f}%")
    print(f"Accuracy: {(100*accuracy_score(np.array(y_teste).argmax(axis=1),np.array(y_prede).argmax(axis=1))):.2f}%")
    print()

cm=confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1))
print("Overall results")
print()
if len(cm)>1:
        if cm[1,0]+cm[1,1] != 0:
            print(f"Recall: {(100*(cm[1,1])/(cm[1,0]+cm[1,1])):.2f}%")
        if cm[0,1]+cm[1,1] != 0:
            print(f"Precision: {(100*(cm[1,1])/(cm[0,1]+cm[1,1])):.2f}%")
print(f"Accuracy: {(100*accuracy_score(np.array(y_test).argmax(axis=1),np.array(y_pred).argmax(axis=1))):.2f}%")