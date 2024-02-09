import numpy as np
import pandas as pd
import pickle as pk
import os

data_file = os.path.join('hosp','patients.csv')
df = pd.read_csv(data_file)

print(len(df))
df=df[(df['anchor_age'].notnull()) & (df['gender'].notnull())]
print('Gender, Age notnull: %d'%len(df))
df=df[df['anchor_age']>=18]
print('#Adults: %d'%len(df))
print(df['anchor_age'].describe())
print(df['gender'].value_counts())