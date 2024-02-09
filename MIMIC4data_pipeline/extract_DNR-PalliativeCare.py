import pandas as pd
import numpy as np
import os
import pickle as pk


admission_file = os.path.join('hosp','admissions.csv')
adm_df = pd.read_csv(admission_file)

Discharge_loc_dict = adm_df['discharge_location'].value_counts().to_dict()
with open(os.path.join('outputs','summary_discharge_locations.csv'),'w') as f:
	f.writelines([k+','+str(Discharge_loc_dict[k])+ '\n' for k in Discharge_loc_dict.keys()])

Discharge_death_file = os.path.join('outputs','Discharge_death.pkl')
Discharge_death_dict = {}

adm_df_death = adm_df[adm_df['discharge_location']=='DIED']
assert(len(adm_df_death)>0)
for hid, grp in adm_df_death.groupby('hadm_id'):
	if len(grp)>1:
		print('Overwritten. %d'%hid)
	Discharge_death_dict[hid] = list(grp['discharge_location'])[0]

print('#stays discharge DEATH: %d'%len(Discharge_death_dict.keys()))
with open(Discharge_death_file,'wb') as f:
	pk.dump(Discharge_death_dict,f)
print('Discharge_death dictionary dumped.')

'''
diagnoses_df:
subject_id,
hadm_id,
seq_num,
icd_code,
icd_version

DNR
V4986(9), Z66(10)

HOSPICE
V667(9), Z51.5(10)
'''
diagnoses_file = os.path.join('hosp','diagnoses_icd.csv')
diagnoses_df = pd.read_csv(diagnoses_file)
print(diagnoses_df.icd_version.value_counts())

diagnoses_df9 = diagnoses_df[diagnoses_df['icd_version']==9]
diagnoses_df10 = diagnoses_df[diagnoses_df['icd_version']==10]
assert(len(diagnoses_df9)>0)
assert(len(diagnoses_df10)>0)

#DNR
DNR_file = os.path.join('outputs','DNR.pkl')
DNR_dict = {}
for hid, grp in diagnoses_df9[diagnoses_df9['icd_code']=='V4986'].groupby('hadm_id'):
	DNR_dict[hid] = list(grp['icd_code'].unique())

for hid, grp in diagnoses_df10[diagnoses_df10['icd_code']=='Z66'].groupby('hadm_id'):
	if hid in DNR_dict.keys():
		DNR_dict[hid].extend(list(grp['icd_code'].unique()))
	else:
		DNR_dict[hid] = list(grp['icd_code'].unique())
print('#stays w. DNR codes: %d'%len(DNR_dict.keys()))

with open(DNR_file,'wb') as f:
	pk.dump(DNR_dict,f)
print('DNR dictionary dumped.')

#Palliative Care + Discharge Hospice
Hospice_file = os.path.join('outputs','Hospice.pkl')
Hospice_dict = {}
for hid, grp in diagnoses_df9[diagnoses_df9['icd_code']=='V667'].groupby('hadm_id'):
	Hospice_dict[hid] = list(grp['icd_code'].unique())

for hid, grp in diagnoses_df10[diagnoses_df10['icd_code']=='Z515'].groupby('hadm_id'):
	if hid in Hospice_dict.keys():
		Hospice_dict[hid].extend(list(grp['icd_code'].unique()))
	else:
		Hospice_dict[hid] = list(grp['icd_code'].unique())
print('#stays w. Hospice codes: %d'%len(Hospice_dict.keys()))


adm_df_hospice = adm_df[adm_df['discharge_location']=='HOSPICE']
print('#stays discharge hospice: %d'%len(adm_df_hospice['hadm_id'].unique()))

for hid in adm_df_hospice['hadm_id'].unique():
	if hid in Hospice_dict.keys():
		Hospice_dict[hid].append('Discharge Hospice')
	else:
		Hospice_dict[hid] = ['Discharge Hospice']

print('Total Hospice #stays: %d'%len(Hospice_dict.keys()))
with open(Hospice_file,'wb') as f:
	pk.dump(Hospice_dict,f)
print('Hospice dictionary dumped.')