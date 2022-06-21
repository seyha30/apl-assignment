import pandas as pd
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
file_path = 'winequality-both.csv'
wine = pd.read_csv(file_path,header=0,sep=',')
print(wine)
