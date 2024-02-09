import numpy as np
import pandas as pd
import pickle as pk
import os

data_file = os.path.join('icu','icustays.csv')
df = pd.read_csv(data_file)

print(len(df))
df=df[(df['stay_id'].notnull()) & (df['subject_id'].notnull())]
print('stayid, pid notnull: %d'%len(df))
df=df[df['los']>=1.0]

print('los>=1: %d'%len(df))
print(df['los'].describe())
print(df['first_careunit'].value_counts())
print(df['last_careunit'].value_counts())

df.to_csv(os.path.join('outputs','icustays-selected.csv'), index = False)