import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
from tqdm import tqdm


'''
GCS names:
GCS-Verbal
GCS-Eye
GCS_Motor

Braden Scale names:
Braden-Sense
Braden-Nut
Braden-Moist
Braden-Mob
Braden-Fric
Braden-Act

Urine Color
Temp Site
'''

GCS_verbal = []
GCS_Eye = []
GCS_Motor = []
Braden_Sense = []
Braden_Nut = []
Braden_Moist = []
Braden_Mob = []
Braden_Fric = []
Braden_Act = []
Urine_Color = []
Temp_Site = []

def extract_unique(df,name):
	res = []
	if len(df[df['newname']==name])>0:
		res = list(df[df['newname']==name].value.unique())
	return res

for sid in tqdm(os.listdir('stays_intermediate')):
	lab_df = pd.read_csv(os.path.join('stays_intermediate',sid,'labs.csv'))
	if len(lab_df)>0:
		GCS_verbal.extend(extract_unique(lab_df, 'GCS-Verbal'))
		GCS_Eye.extend(extract_unique(lab_df, 'GCS-Eye'))
		GCS_Motor.extend(extract_unique(lab_df, 'GCS_Motor'))
		Braden_Sense.extend(extract_unique(lab_df, 'Braden-Sense'))
		Braden_Nut.extend(extract_unique(lab_df, 'Braden-Nut'))
		Braden_Moist.extend(extract_unique(lab_df, 'Braden-Moist'))
		Braden_Mob.extend(extract_unique(lab_df, 'Braden-Mob'))
		Braden_Fric.extend(extract_unique(lab_df, 'Braden-Fric'))
		Braden_Act.extend(extract_unique(lab_df, 'Braden-Act'))
		Urine_Color.extend(extract_unique(lab_df, 'Urine Color'))
		Temp_Site.extend(extract_unique(lab_df, 'Temp Site'))

	chart_df = pd.read_csv(os.path.join('stays_intermediate',sid,'charts.csv'))
	if len(chart_df)>0:
		GCS_verbal.extend(extract_unique(chart_df, 'GCS-Verbal'))
		GCS_Eye.extend(extract_unique(chart_df, 'GCS-Eye'))
		GCS_Motor.extend(extract_unique(chart_df, 'GCS_Motor'))
		Braden_Sense.extend(extract_unique(chart_df, 'Braden-Sense'))
		Braden_Nut.extend(extract_unique(chart_df, 'Braden-Nut'))
		Braden_Moist.extend(extract_unique(chart_df, 'Braden-Moist'))
		Braden_Mob.extend(extract_unique(chart_df, 'Braden-Mob'))
		Braden_Fric.extend(extract_unique(chart_df, 'Braden-Fric'))
		Braden_Act.extend(extract_unique(chart_df, 'Braden-Act'))
		Urine_Color.extend(extract_unique(chart_df, 'Urine Color'))
		Temp_Site.extend(extract_unique(chart_df, 'Temp Site'))

GCS_verbal = pd.Series(GCS_verbal)
GCS_Eye = pd.Series(GCS_Eye)
GCS_Motor = pd.Series(GCS_Motor)
Braden_Sense = pd.Series(Braden_Sense)
Braden_Nut = pd.Series(Braden_Nut)
Braden_Moist = pd.Series(Braden_Moist)
Braden_Mob = pd.Series(Braden_Mob)
Braden_Fric = pd.Series(Braden_Fric)
Braden_Act = pd.Series(Braden_Act)
Urine_Color = pd.Series(Urine_Color)
Temp_Site = pd.Series(Temp_Site )

print('GCS_verbal')
print(GCS_verbal.value_counts())
print('GCS_Eye')
print(GCS_Eye.value_counts())
print('GCS_Motor')
print(GCS_Motor.value_counts())
print('Braden_Sense')
print(Braden_Sense.value_counts())
print('Braden_Nut')
print(Braden_Nut.value_counts())
print('Braden_Moist')
print(Braden_Moist.value_counts())
print('Braden_Mob')
print(Braden_Mob.value_counts())
print('Braden_Fric')
print(Braden_Fric.value_counts())
print('Braden_Act')
print(Braden_Act.value_counts())
print('Urine_Color')
print(Urine_Color.value_counts())
print('Temp_Site')
print(Temp_Site.value_counts())
