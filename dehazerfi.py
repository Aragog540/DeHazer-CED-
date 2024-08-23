# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 22:58:50 2024

@author: swaro
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Input,UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model

image_names = os.listdir(r"C:\Users\swaro\Desktop\dehazer\Dense_Haze_NTIRE19\hazy")
folder_path = r"C:\Users\swaro\Desktop\dehazer\Dense_Haze_NTIRE19\hazy"
image_names.sort()
x=[]
for image_file in image_names:
    image_path = os.path.join(folder_path, image_file)
    arr=cv2.imread(image_path)
    arr2=cv2.resize(arr,(256,256),interpolation=cv2.INTER_LINEAR)
    x.append(arr2)
image_names = os.listdir(r"C:\Users\swaro\Desktop\dehazer\Dense_Haze_NTIRE19\clear")
folder_path = r"C:\Users\swaro\Desktop\dehazer\Dense_Haze_NTIRE19\clear"
image_names.sort()
y=[]
for image_file in image_names:
    image_path = os.path.join(folder_path, image_file)
    arr=cv2.imread(image_path)
    arr2=cv2.resize(arr,(256,256),interpolation=cv2.INTER_LINEAR)
    y.append(arr2)
    
x=np.array(x)/255
y=np.array(y)/255
x[0].shape

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.5)    

xtrain=np.array(xtrain)
ytrain=np.array(ytrain)
xtest=np.array(xtest)
ytest=np.array(ytest)
    




inputs=Input(shape=(256,256,3))

con1=Conv2D(32,5,activation='relu',padding='same')(inputs)
maxp=MaxPooling2D(2,padding='same')(con1)
bn=BatchNormalization()(maxp)


con2=Conv2D(64,5,activation='relu',padding='same')(bn)
maxp1=MaxPooling2D(2,padding='same')(con2)
bn=BatchNormalization()(maxp1)


con3=Conv2D(80,5,activation='relu',padding='same')(bn)
maxp2=MaxPooling2D(2,padding='same')(con3)
bn=BatchNormalization()(maxp2)


con3=Conv2D(128,5,activation='relu',padding='same')(bn)
maxp2=MaxPooling2D(2,padding='same')(con3)
bn=BatchNormalization()(maxp2)


con3=Conv2D(256,5,activation='relu',padding='same')(bn)

maxp2=MaxPooling2D(2,padding='same')(con3)
bn=BatchNormalization()(maxp2)



con4=Conv2D(256,5,activation='relu',padding='same')(bn)
con4=BatchNormalization()(con4)
up=UpSampling2D(2)(con4)


con4=Conv2D(128,5,activation='relu',padding='same')(up)
con4=BatchNormalization()(con4)
up=UpSampling2D(2)(con4)


con4=Conv2D(80,5,activation='relu',padding='same')(up)
con4=BatchNormalization()(con4)
up=UpSampling2D(2)(con4)


con5=Conv2D(64,5,activation='relu',padding='same')(up)
con5=BatchNormalization()(con5)
up1=UpSampling2D(2)(con5)


con6=Conv2D(32,5,activation='relu',padding='same')(up1)
con6=BatchNormalization()(con6)
up2=UpSampling2D(2)(con6)


out=Conv2D(3,4,activation='sigmoid',padding='same')(up2)


model=Model(inputs=inputs,outputs=out)
model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
histr=model.fit(x,y,epochs=1000,validation_data=[xtest,ytest],batch_size=5)
model.save('modelok.keras')
