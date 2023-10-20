#本代码要求输入一个已经经过特征编码后的文件，并提供训练好的模型目录，该脚本将读入输入的序列文件并对其中的每条序列进行预测，输出最终的预测结果
import os
import timeit
import datetime
import math
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model, model_from_json
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import GlobalAvgPool3D, GlobalMaxPool3D
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.layers import concatenate, add, multiply, average, maximum
from tensorflow.keras.layers import BatchNormalization, Activation,concatenate
from keras.layers.embeddings import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix,f1_score
from tensorflow.keras import optimizers,regularizers
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from pickle import load
import sys
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import os
from matplotlib import pyplot as plt
import numpy as np
import math
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#读取输入的序列文件，文件格式要求，第一列为序列名称，第二列及以后为编码后的序列特征
def read_file(X):
    File = pd.read_csv(str(X), header=None)
    File = np.array(File)
    File_X=File[:,1:]
    File_item_name=File[:,0]
    File_X=np.array(File_X)
    File_item_name=np.array(File_item_name)
    File_X=File_X.astype('float64')
    return File_X,File_item_name
#对输入的特征进行0-1标准化
def normal_phy(Arr):
    max = np.max(Arr,axis=0)
    min = np.min(Arr,axis=0)
    Arr -=min
    Arr /=(max-min)
    return Arr
#The data for each item should be [1,1904]
def sigmoid_function(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z[i][j]=(1/(1 + math.exp(-z[i][j])))
    return z
#处理输入的序列
def Trans_data(Array):
    print(Array.shape)
    Unirep_array=[]
    PHY_array=[]
    Unirep_arr=Array[:,0:1900]
    Unirep_arr=normal_phy(Unirep_arr)
    Phy_arr=Array[:,1900:1904]
    PHY_arr=normal_phy(Phy_arr)
    print(Array.shape)
    print(Unirep_arr.shape)
    for i in range(len(Unirep_arr)):
        sort_array=np.reshape(Unirep_arr[i],(1,1900))
        Unirep_array.append(sort_array)
    Unirep_array=np.array(Unirep_array)
    Unirep_array=np.reshape(Unirep_array,(-1,38,25,2))
    Unirep_array=Unirep_array.astype('float64')  
    for i in range(len(PHY_arr)):
        sort_array=np.reshape(PHY_arr[i],(1,4))
        PHY_array.append(sort_array)
    PHY_array=np.array(PHY_array)
    PHY_array=np.reshape(PHY_array,(-1,2,2,1))
    PHY_array=PHY_array.astype('float64')
    return Unirep_array,PHY_array
input_path="XXXXXXX"
Data,item_name=read_file("input_path")
print(item_name.shape,Data.shape)
TAPE,PHY=Trans_data(Data)
PHY=np.reshape(PHY,(-1,2, 2, 1))
item_name=np.array(item_name)
TAPE=TAPE.astype('float64')
PHY=PHY.astype('float64')
##Set the model path
Model_dir="XXXX"
HOME_DIR = Model_dir
os.chdir(Model_dir)
json_file = open('./Unirep_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_Unirep = model_from_json(best_model_json)
best_model_Unirep.load_weights("./Unirep_best_weights.hdf5")
print("Loaded best Unirep Model weights from disk")
Unirep_h=Model(inputs=best_model_Unirep.input,outputs=best_model_Unirep.layers[-2].output)
Unirep_h_fea= Unirep_h.predict(TAPE)
Unirep_h_fea=np.array(Unirep_h_fea)
json_file = open('./PHY_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_PHY = model_from_json(best_model_json)
best_model_PHY.load_weights("./PHY_best_weights.hdf5")
print("Loaded best Unirep Model weights from disk")
PHY_h=Model(inputs=best_model_PHY.input,outputs=best_model_PHY.layers[-2].output)
PHY_h_fea= PHY_h.predict(PHY)
PHY_h_fea=np.array(PHY_h_fea)
json_file = open('./Last_classifier.json', 'r')
best_model_json = json_file.read()
json_file.close()
best_model_MIX = model_from_json(best_model_json)
best_model_MIX.load_weights("./MIX_best_weights.hdf5")
print("Loaded best MIX Model weights from disk")
Total_Fea=np.c_[Unirep_h_fea,PHY_h_fea]
Total_Fea=np.reshape(Total_Fea,(-1,1,128,1))
Y_pred = best_model_MIX.predict(Total_Fea)
lable=[]
for item in Y_pred:
    lable.append(np.argmax(item))
lable=np.array(lable)
predict=np.c_[item_name,lable,Y_pred]
with open("./predict.txt", "w") as f3:
    np.savetxt(f3,predict,fmt="%s")
