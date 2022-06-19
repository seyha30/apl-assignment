import pandas as pd
import glob as glb
import os
all_files = glb.glob(os.path.join('../','*.csv'))
# clear previous file content
file = open('winequality-both.csv','w+')
file.close()
for file_name in all_files:
    data_frame = pd.read_csv(file_name,sep=',',header=0)
    # split wine type
    type = file_name.split('-')[1].split('.')[0]
    if type == 'red':
        data_frame.insert(0,'type',type)
    if type == 'white':
        data_frame.insert(0,'type',type)
    # write to csv
    data_frame.to_csv('winequality-both.csv',mode='a',index=False)
