'''
进阶循环神经网络用法
1.Recurrent dropout
2.Stacking recurrent layers
3.Bidirectional recurrent layers
'''

import os
import numpy as np
from matplotlib import pyplot as plt

fname='E:/DataSet/jena_climate/jena_climate_2009_2016.csv'
f=open(fname)
data=f.read()
f.close()

lines=data.split('\n')     #数据行数
header=lines[0].split(',') #属性名

#Parsing the data
float_data=np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines): #用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values

#Plotting the temperature timeseries
temp=float_data[:,1]
plt.plot(range(len(temp),temp))

#Plotting the first 10 days of the temperature timeseries
plt.plot(range(1440),temp[:1440]) #data recorded every 10 mins

