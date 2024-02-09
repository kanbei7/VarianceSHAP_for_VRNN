import pandas as pd
import numpy as np
import pickle as pk
import os
import sys

'''
224385	Intubation	intubation
225448	Percutaneous Tracheostomy	intubation
226237	Open Tracheostomy	intubation

227194	Extubation	extubation
225468	Unplanned Extubation (patient-initiated)	extubation
225477	Unplanned Extubation (non-patient initiated)	extubation

225792	Invasive Ventilation	ventilator-invasive
225794	Non-invasive Ventilation	Ventilator-noninvasive

'''
ITEM_SET = [224385,225448,226237,225477,225468,227194,225792,225794]
Intubation = [224385,225448,226237]
Extubation = [225477,225468,227194]
Ventilator_invasive = [225792]
Ventilator_noninvasive = [225794]

df = pd.read_csv(os.path.join('icu','procedureevents.csv'))
df = df[df['itemid'].isin(ITEM_SET)]
print('N records: %d'%len(df))
print('N stays: %d'%len(df.stay_id.unique()))

df.to_csv(os.path.join('outputs','ventilation.csv'),index = False)

intub_df = df[df['itemid'].isin(Intubation)]
extub_df = df[df['itemid'].isin(Extubation)]
vent_inv_df = df[df['itemid'].isin(Ventilator_invasive)]
vent_noninv_df = df[df['itemid'].isin(Ventilator_noninvasive)]

A = intub_df.stay_id.unique()
B = extub_df.stay_id.unique()
C = vent_inv_df.stay_id.unique()
D = vent_noninv_df.stay_id.unique()

print('intub #stays: %d'%len(A))
print('extub #stays: %d'%len(B))
print('vent-inv #stays: %d'%len(C))
print('vent-noninv #stays: %d'%len(D))

print('intub but not vent-inv(#stays): %d'%len([sid for sid in A if not sid in C]))
print('extub but no intub records(#stays): %d'%len([sid for sid in B if not sid in A]))
print('extub but no intub/vent-inv records(#stays): %d'%len([sid for sid in B if (not sid in A) and (not sid in C)]))
print('extub but no intub/vent-inv/vent-noninv records(#stays): %d'%len([sid for sid in B if (not sid in A) and (not sid in C) and (not sid in D)]))