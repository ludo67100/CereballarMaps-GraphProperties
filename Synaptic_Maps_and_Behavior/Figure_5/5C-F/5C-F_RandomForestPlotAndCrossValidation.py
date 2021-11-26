# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 16:25:41 2020

@author: ludovic.spaeth
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
import pandas as pd
import numpy as np  
import seaborn as sn
from matplotlib import pyplot as plt
from scipy import stats
import os


alpha=0.5
swarm = True
sharex, sharey = True, True
cap=0.2


fig, ax = plt.subplots(1,2, figsize=(12,4))

#First the synaptic properties 

file = 'D:/000_PAPER/00_ANSWER_TO_REVIEWERS/RandomForestanalysis/OuputWithFolds/RF_Output_With_Folds_Training_Segregated_SORTED_LABELS.xlsx'
rdnFile = 'D:/000_PAPER/00_ANSWER_TO_REVIEWERS/RandomForestanalysis/OuputWithFolds/RF_Output_With_Folds_Training_Segregated_RANDOM_LABELS.xlsx'


df = pd.read_excel(file,header=0)
rdnDf = pd.read_excel(rdnFile, header=0)


#result = df.groupby('Random Forest').mean()[['Random Forest (CTRL)','Random Forest (EC)',
#                                             'Random Forest (ENR)','Random Forest (ES)',
#                                             'Random Forest (LC)','Random Forest (LS)']]

sortedResult = df.groupby('Condition').mean()[['Random Forest (CTRL)','Random Forest (EC)',
                                               'Random Forest (LTR)','Random Forest (STR)','Random Forest (ES)',
                                               'Random Forest (LC)','Random Forest (LS)']]

rndmResult = rdnDf.groupby('Condition').mean()[['Random Forest (CTRL)','Random Forest (EC)',
                                                'Random Forest (LTR)','Random Forest (STR)','Random Forest (ES)',
                                                'Random Forest (LC)','Random Forest (LS)']]




sortedResult = sortedResult.reindex(index=['EC', 'LTR','STR', 'ES', 'LC', 'LS', 'CTRL'], columns=['Random Forest (EC)','Random Forest (LTR)','Random Forest (STR)',
                                                                                                  'Random Forest (ES)','Random Forest (LC)',
                                                                                                  'Random Forest (LS)','Random Forest (CTRL)'])

rndmResult = rndmResult.reindex(index=['EC', 'LTR','STR', 'ES', 'LC', 'LS', 'CTRL'], columns=['Random Forest (EC)','Random Forest (LTR)','Random Forest (STR)',
                                                                                              'Random Forest (ES)','Random Forest (LC)',
                                                                                              'Random Forest (LS)','Random Forest (CTRL)'])


#Show heatmaps
sn.heatmap(sortedResult,annot=True, cmap='magma_r', vmax=0.5, ax=ax[0])
ax[0].set_title('Sorted Lables (synaptic)')

sn.heatmap(rndmResult, annot=True, cmap='magma_r', vmax=0.5, ax=ax[1])
ax[1].set_title('Random Lables (synaptic)')



ACCURACIES, RDM_ACCURACIES = [],[]
#Determine accuracy for sorted labels
for condition in np.unique(df['Condition'].values):
    
    subDf = df.loc[df['Condition']==condition]
    avg = subDf.groupby('Fold').mean()
    accuracyInCondition = avg['Random Forest ({})'.format(condition)].values
    ACCURACIES.append(accuracyInCondition)
    
#Determine accuracy for randomized labels
for condition in np.unique(df['Condition'].values):
    
    subDf = rdnDf.loc[rdnDf['Condition']==condition]
    avg = subDf.groupby('Fold').mean()
    accuracyInCondition = avg['Random Forest ({})'.format(condition)].values
    RDM_ACCURACIES.append(accuracyInCondition)
    
    
sortedLabelsAccuracy = pd.DataFrame(ACCURACIES, index=np.unique(df['Condition'].values)).T.mean(axis=1)
randomLabelAccuracy = pd.DataFrame(RDM_ACCURACIES, index=np.unique(rdnDf['Condition'].values)).T.mean(axis=1)


barPlotdf = pd.concat([sortedLabelsAccuracy,randomLabelAccuracy],axis=1)
barPlotdf.columns = ['Sorted Labels', 'Random Labels']

fig2, axx = plt.subplots(1,5,sharex=sharex, sharey=sharex, figsize=(16,5))
axx[0].set_ylabel('Accuracy')

if swarm == True:
    sn.violinplot(data=barPlotdf, color='0.8', ax=axx[0], inner='quart', alpha=alpha)
    
sn.pointplot(data=barPlotdf, ax=axx[0], capsize=cap,ci=None)

#And now, stats
#First normality 
print ("[1] Accuracy on Synaptic params")
normPval = []
for dist, idx in zip([sortedLabelsAccuracy.values, randomLabelAccuracy.values], ['Sorted accuracy','Random accuracy']): 
    
    print ('{} avg accuracy = {}+/-{}'.format(idx,np.mean(dist),np.std(dist)))
    
    normPval.append(stats.shapiro(dist)[1])

if normPval[0] >= 0.05 and normPval[1] >=0.05: 
    print ('Both distributions are normal')
    
    test = stats.ttest_ind(sortedLabelsAccuracy.values, randomLabelAccuracy.values)
    testType = 't-test'
    print ('T-test random vs sorted cross validation | stat = {} ; p = {}'.format(test[0], test[1]))
    
else: 
    print ('At least one of the distribution is not normal')
    test = stats.mannwhitneyu(sortedLabelsAccuracy.values, randomLabelAccuracy.values)
    testType = 'MWU'
    print('MWU random vs sorted cross validation | stat = {} ; p = {}'.format(test[0], test[1]))
    
axx[0].set_title('Synaptic Params \n {} p_val={:.3E}'.format(testType,test[1]))

#Do a simple df with accuracy for synaptic params

synAccDf = pd.DataFrame([sortedLabelsAccuracy.values,randomLabelAccuracy.values],index=['actual','shuffled']).T
synAccDf.to_excel('D:/000_PAPER/00_ANSWER_TO_REVIEWERS/RandomForestanalysis/SynapticParamsAccuracyDists.xlsx')


#TODO : implement Jyotika's cross validation 
#The file containing the csv's
sourceFolderJB = 'D:/000_PAPER/00_ANSWER_TO_REVIEWERS/RandomForestanalysis/GraphPropFolds'
print ('[2] Accuracy on Graph Properties')

for file, index in zip(os.listdir(sourceFolderJB), range(len(os.listdir(sourceFolderJB)))):

    print()
    print(file)
    graphDf = pd.read_csv('{}/{}'.format(sourceFolderJB, file), delimiter=',')
    

    if swarm==True:
        sn.violinplot(x='Labels',y='Accuracy', data=graphDf, ax=axx[index+1],inner='quart', color='0.8', alpha=alpha)
        
    sn.pointplot(x='Labels',y='Accuracy', data=graphDf, ax=axx[index+1],capsize=cap,ci=None)
        
    #And now the stats
    normPval = []
    distributions = []
    for dist, idx in zip([graphDf.loc[graphDf['Labels']=='Sorted Labels']['Accuracy'].values, 
                          graphDf.loc[graphDf['Labels']=='Random Labels']['Accuracy'].values],
                          ['Sorted accuracy','Random accuracy']): 
        
        print ('{} avg accuracy = {}+/-{}'.format(idx,np.mean(dist),np.std(dist)))
        
        normPval.append(stats.shapiro(dist)[1])
        distributions.append(dist)
    
    if normPval[0] >= 0.05 and normPval[1] >=0.05: 
        print ('Both distributions are normal')
        
        test = stats.ttest_ind(distributions[0], distributions[1])
        testType = 't-test'
        print ('T-test random vs sorted cross validation | stat = {} ; p = {}'.format(test[0], test[1]))
        
    else: 
        print ('At least one of the distribution is not normal')
        test = stats.mannwhitneyu(distributions[0], distributions[1], alternative='two-sided')
        testType = 'MWU'
        print('MWU random vs sorted cross validation | stat = {} ; p = {}'.format(test[0], test[1]))
        
    axx[index+1].set_title('Graph Properties {}_{} \n {} p_val={:.3E}'.format(file.split('_')[2],file.split('_')[3],testType,test[1]))
    
    
plt.tight_layout()
    
    

