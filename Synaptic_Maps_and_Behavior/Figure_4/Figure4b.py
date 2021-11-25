# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:33:31 2021

Computes panel 4b from dataset 

@author: klab
"""



#Indicate path of DataSet
datadir = 'C:/Users/klab/Documents/SpaethBahugunaData'
#Where to save the data
saveDir = 'C:/Users/klab/Desktop/testOutput'

#---------------------------------------the code-------------------------------------
#------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------

import matplotlib.pyplot as plt 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 7})
import os 
import numpy as np
import math
import pandas as pd 
import seaborn as sn
import scipy as sp

dataDir = '{}/ProcessedData/Adaptive_Dataset'.format(datadir)

conditions = os.listdir(dataDir)

MANIPLABEL, CONDITION, PROPORTION = [],[],[]

dfTempProportion, dfTempAverage = [],[]

for i in range(len(conditions)):
    
    parentDir = '{}/{}'.format(dataDir, conditions[i])
    
    proportion, averageAmp, labels = [],[],[]
    
    for manip in os.listdir(parentDir): 
        
        manipPath = '{}/{}'.format(parentDir, manip)
        
        zScore2D = np.genfromtxt('{}/{}_Amp_zscore_2D_OK.csv'.format(manipPath, manip), delimiter=',')
        flatZscore = np.ravel(zScore2D)
        
        #Get proportion of active sites 
        cleanArray = [x for x in flatZscore if math.isnan(x)==False]
        prop = len([x for x in cleanArray if x >= 3.09])/len(cleanArray)*100
        
        averageAmp.append(np.nanmean([x for x in cleanArray if x >= 3.09]))
        proportion.append(prop)
        MANIPLABEL.append(manip)
        CONDITION.append(conditions[i])
        PROPORTION.append(prop)
        
    dfTempProportion.append(proportion)
    dfTempAverage.append(averageAmp)
    
    #Values
    print(conditions[i])
    print('n={}'.format(len(proportion)))
    print('Proportion of active sites, avg+/-SD: {} ({})'.format(round(np.nanmean(proportion),2), round(np.nanstd(proportion),2)))
    print()
    

df = pd.DataFrame(dfTempProportion, index=conditions).T
dfAmp = pd.DataFrame(dfTempAverage, index=conditions).T

#Figure-------------------------------------------------

fig, ax = plt.subplots(1,1)    

ax.set_ylabel('Proportion of active sites (%)')
sn.boxplot(data=df, showfliers=False, order=['WT','ENR1','ENR2','ES','EC','LS','LC'], ax=ax)
sn.swarmplot(data=df,color='black',order=['WT','ENR1','ENR2','ES','EC','LS','LC'], ax=ax)
Propkruskal = sp.stats.kruskal(dfTempProportion[0],
                               dfTempProportion[1],
                               dfTempProportion[2],
                               dfTempProportion[3],
                               dfTempProportion[4],
                               dfTempProportion[5],
                               dfTempProportion[6])

print('Kruskal Wallis: Proportion of active sites')
print(Propkruskal)



#Do global Df for datasource
globDf = pd.DataFrame([MANIPLABEL, CONDITION, PROPORTION], index=['Experiment', 'Condition', 'Proportion of Active sites (%)']).T



        
        
        
        