from importlib.resources import path
from operator import index
import pandas as pd
import numpy as np
import seaborn as sbn
import glob as glb
import os
from matplotlib import pyplot as plt
all_files = glb.glob(os.path.join('../','*.csv'))
# clear previous file content
file = open('winequality-both.csv','w+')
file.close()
for file_name in all_files:
    data_frame = pd.read_csv(file_name,sep=',',header=0)
    # print(file_name)
    type = file_name.split('-')[1].split('.')[0]
    if type == 'red':
        data_frame.insert(0,'type','red')
    if type == 'white':
        data_frame.insert(0,'type','white')
    data_frame.to_csv('winequality-both.csv',mode='a',index=False)
    # print(data_frame.describe())
