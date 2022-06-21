import pandas as pd
import glob as glb
import os
all_files = glb.glob(os.path.join('../','*.csv'))
# clear previous file content
file = open('winequality-both.csv','w+')
file.close()
temporary_data_frame = []
for file_name in all_files:
    data_frame = pd.read_csv(file_name)
    # split wine type
    type = file_name.split('-')[1].split('.')[0]
    if type == 'red':
        data_frame.insert(0,'type',type)
    if type == 'white':
        data_frame.insert(0,'type',type)
    temporary_data_frame.append(data_frame)
    # write to csv
df = pd.concat(temporary_data_frame)
df.to_csv('winequality-both.csv',mode='a',index=False)
