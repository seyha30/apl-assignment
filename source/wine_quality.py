from importlib.resources import path
import pandas as pd
import numpy as np
import seaborn as sbn
import glob as glb
import os
from matplotlib import pyplot as plt
all_files = glb.glob(os.path.join('../','*.csv'))
for file_name in all_files:
    data_frame = pd.read_csv(file_name,sep=',',header=0)
    data_frame.to_csv('winequality-both.csv',index=False)
