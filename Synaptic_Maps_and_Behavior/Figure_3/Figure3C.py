# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:03:46 2020

Computes balance profiles of figure 3C from DataSource_Spaeth_Bahuguna_et_al.xlsx'

@author: Ludovic.spaeth
"""




#Indicate path of DataSource_Spaeth_Bahuguna_et_al.xlsx
file = 'D:/000_PAPER/000/DataSource_Spaeth_Bahuguna_et_al.xlsx'
#Where to save the data
savedir = 'C:/Users/klab/Desktop/testOutput'

#---------------------------------------the code-------------------------------------
#------------------------------------------------------------------------------------

import matplotlib.pyplot as plt 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 7})
import pandas as pd 
import matplotlib.pyplot as plt


dataset = pd.read_excel(file,header=28,index_col=0,sheet_name='Figure 3c')


fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].set_ylabel('Raw Balance Index (mean+/-SEM)') ; ax[1].set_ylabel('Balance Index (norm. to baseline, mean+/-SEM)')

#Raw profiles------------------------------------------------------------------
average = dataset.groupby('Condition').mean()
sem = dataset.groupby('Condition').sem()

for group in average.index:
    
    ax[0].plot(average.loc[group],label=group)
    ax[0].fill_between([i for i in range(average.columns.size)],
                       average.loc[group]+sem.loc[group],
                       average.loc[group]-sem.loc[group],
                       alpha=0.3)

#Normalize to baseline---------------------------------------------------------
for index in dataset.index:
    
    if dataset.loc[index,'baseline'] > 0:
        
        dataset.loc[index,'baseline':'day 33'] = dataset.loc[index,'baseline':'day 33']-dataset.loc[index,'baseline']
        
    else:
        
        dataset.loc[index,'baseline':'day 33'] = dataset.loc[index,'baseline':'day 33']+abs(dataset.loc[index,'baseline'])
        
        
average = dataset.groupby('Condition').mean()
sem = dataset.groupby('Condition').sem()

for group in average.index:
    
    ax[1].plot(average.loc[group],label=group)
    ax[1].fill_between([i for i in range(average.columns.size)],
                       average.loc[group]+sem.loc[group],
                       average.loc[group]-sem.loc[group],
                       alpha=0.3)

    
ax[0].legend(loc='best'); ax[1].legend(loc='best')


