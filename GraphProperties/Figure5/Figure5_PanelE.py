import os
import glob
import numpy as np
import pylab as pl
import scipy.io as sio
# for_Jyotika.m
from copy import copy, deepcopy
import pickle
import matplotlib.cm as cm
import pdb
import h5py
import pandas as pd
import scipy.stats as sp_st
import sys
import seaborn as sns
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels as sms
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import roc_curve, auc,confusion_matrix
from sklearn.ensemble import RandomForestClassifier


sys.path.append("common/")
import analyze as anal

data_target_dir = "./data/"
fig_target_dir = "./Figure5/"

ENR_7 = ["180517","190517","160517","170517","200517","270820","280820"]
ENR_19 = ["220319","230319","210319","171219","090920","100920"]



zone_names = ["B_contra","AX_contra","Alat_contra","Amed_contra","Amed_ipsi","Alat_ipsi","AX_ipsi","B_ipsi"]

labels =  ['EC', 'ES', 'S-TR', 'L-TR', 'LC', 'LS', 'WT']


ipsi_contra = sys.argv[1]
st_type = sys.argv[2]

if ipsi_contra == "n":
    independent_props = ["modularity_index","participation_pos","module_degree_zscore","local_assortativity_pos_whole"]
elif ipsi_contra == "y":
    independent_props = ["modularity_index", "participation_pos_ipsi","module_degree_zscore_ipsi","participation_pos_contra","module_degree_zscore_contra","local_assortativity_pos_ipsi","local_assortativity_pos_contra"]

elif ipsi_contra == "zone_wise":
    independent_props = [y+"-"+x  for x in ["module_degree_zscore","participation_pos","local_assortativity_pos_whole"] for y in zone_names ] 

def avg_over_gammas_and_animals(field,data,non_string_columns,metric='mean'):
    temp_dat1_wgp = dict()
    for k in non_string_columns:
        temp_dat1_wgp[k] = []

    if metric == "mean":
        sub_dat1_wgp = anal.mean_over_gammas(field,data,temp_dat1_wgp,non_string_columns)
    elif metric == "median":
        sub_dat1_wgp = anal.median_over_gammas(field,data,temp_dat1_wgp,non_string_columns)
    for k in non_string_columns:
        sub_dat1_wgp[k] = sub_dat1_wgp[k].astype('float')

    return sub_dat1_wgp



def fill_confusion_matrix(clf,conf_mat_list,X_train,Y_train,X_test,Y_test,tree_depth_list,train_labels,test_labels):
    clf.fit(X_train, Y_train) 
    #tree_depth = [ estimator.tree_.max_depth for estimator in clf.estimators_]
    #print(tree_depth)
    #tree_depth_list.append(tree_depth)
    y_pred = clf.predict(X_test) 
    confusion_mat = pd.crosstab(Y_test, y_pred,rownames=["Actual"],colnames=["Predicted"])
 
    print(confusion_mat) 
    confusion_mat = confusion_mat.reindex(index=test_labels,columns=train_labels)
    print(confusion_mat) 
    conf_mat_list.append(confusion_mat.div(confusion_mat.sum(axis=1),axis=0))




if ipsi_contra == "n":
    graph_prop_enr_behav_whole = pd.read_csv(data_target_dir+"graph_properties_with_behavior_pandas_all.csv") # From read_data_behavior.py, remove _all.csv there to revert to without seed graph properties
elif ipsi_contra == "y":

    graph_prop_enr_behav_whole = pd.read_csv(data_target_dir+"graph_properties_with_behavior_pandas_sub_ipsi_contra_all.csv") # From read_data_behavior.py, remove _all.csv there to revert to without seed graph properties
elif ipsi_contra == "zone_wise":
    graph_prop_enr_behav_whole = pd.read_csv(data_target_dir+"merged_zone_wise_graph_props_avg_seeds.csv")
elif ipsi_contra == "semi_zone_wise": # Instead of values for every zone, only use the mean and variabnce of the zone wise trajectory in order zone_names
    graph_prop_enr_behav_whole = pd.read_csv(data_target_dir+"merged_zone_wise_graph_props_avg_seeds.csv")
    independent_props = []
    for x in ["module_degree_zscore","participation_pos","local_assortativity_pos_whole"]:
        temp_prop = [y+"-"+x  for y in zone_names]
        sub_data = graph_prop_enr_behav_whole[temp_prop]
        mean = np.mean(np.array(sub_data),axis=1)
        var = np.var(np.array(sub_data),axis=1)
        independent_props.append(x+"-"+"mean")
        graph_prop_enr_behav_whole[x+"-"+"mean"] = mean
        independent_props.append(x+"-"+"var")
        graph_prop_enr_behav_whole[x+"-"+"var"] = var


print(len(graph_prop_enr_behav_whole))

graph_prop_enr_behav_whole["subtype"] = graph_prop_enr_behav_whole["subtype"].replace('ENR1','S-TR')
graph_prop_enr_behav_whole["subtype"] = graph_prop_enr_behav_whole["subtype"].replace('ENR2','L-TR')



dat_wgp = graph_prop_enr_behav_whole
dat_wgp["short-names"] = [ x.split('-')[0]+"-"+x.split('-')[1]  for x in dat_wgp["names"] ]

non_string_columns = list(set(dat_wgp.keys())-set(['Unnamed: 0', 'Unnamed: 0.1','names', 'subtype',"short-names","seed","gamma"]))
sub_dat_wgp = avg_over_gammas_and_animals(["names","gamma"],dat_wgp,non_string_columns,"median") # Median over gammas
sub_dat_wgp["short-names"] = [ x.split('-')[0]+"-"+x.split('-')[1]  for x in sub_dat_wgp["names"] ]

sub_dat1_wgp = avg_over_gammas_and_animals(["names"],sub_dat_wgp,non_string_columns,"mean") # Mean over cells

subtypes = np.zeros(len(sub_dat1_wgp["subtype"]))
for i,st in enumerate(np.unique(sub_dat1_wgp["subtype"])):
    ind_lc = np.where(np.array(sub_dat1_wgp["subtype"])==st)[0]
    subtypes[ind_lc] = i
sub_dat1_wgp["subtype_val"] = subtypes


if "zone" in ipsi_contra:
    sub_dat11_wgp = sub_dat1_wgp[independent_props+["subtype","subtype_val","names","modularity_index"]]
else:
    sub_dat11_wgp = sub_dat1_wgp[independent_props+["subtype","subtype_val","names"]]
sub_dat11_wgp["short_names"] = [ x.split('-')[0]  for x in sub_dat11_wgp["names"]]


sub_dat11_wgp = sub_dat11_wgp.dropna()

clf = RandomForestClassifier(class_weight='balanced',max_depth=30,criterion='entropy',n_estimators=200)
clf_shuff = RandomForestClassifier(class_weight='balanced',max_depth=30,criterion='entropy',n_estimators=200)
n_splits = 250
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
X = sub_dat11_wgp[independent_props]
Y = sub_dat11_wgp[st_type]
Y_shuff = np.random.permutation(Y)

confusion_mats_all = []
score_all = []
tree_depth_list=[]
tree_depth_list_shuff = []
confusion_mats_all_shuff = []
score_shuff_all = []
for (train, test), i in zip(sss.split(X, Y), range(n_splits)): 
    fill_confusion_matrix(clf,confusion_mats_all,X.iloc[train],Y.iloc[train],X.iloc[test],Y.iloc[test],tree_depth_list,labels,labels)
    fill_confusion_matrix(clf_shuff,confusion_mats_all_shuff,X.iloc[train],Y_shuff[train],X.iloc[test],Y_shuff[test],tree_depth_list_shuff,labels,labels)
    score_all.append(clf.score(X.iloc[test],Y.iloc[test]))
    score_shuff_all.append(clf_shuff.score(X.iloc[test],Y_shuff[test]))


accuracy_comparison_df = pd.DataFrame(columns=["Accuracy","Labels"])
acc_temp = dict()
for k in accuracy_comparison_df.keys():
	acc_temp[k] = []

acc_temp["Accuracy"].append(np.hstack(score_all))
acc_temp["Labels"].append([ "Sorted Labels" for i in np.arange(len(np.hstack(score_all)))])

acc_temp["Accuracy"].append(np.hstack(score_shuff_all))
acc_temp["Labels"].append([ "Random Labels" for i in np.arange(len(np.hstack(score_shuff_all)))])

for k in acc_temp.keys():
	accuracy_comparison_df[k] = np.hstack(acc_temp[k])


accuracy_comparison_df.to_csv(data_target_dir+"accuracy_comparison_"+ipsi_contra+"_"+st_type+".csv")

if ipsi_contra == "n":
    tit1 = "Whole graph properties"
elif ipsi_contra == "y":
    tit1 = "Ipsi/Contra graph properties"
elif ipsi_contra == "zone_wise":
    tit1 = "Zone wise graph properties"
elif ipsi_contra == "semi_zone_wise":
    tit1 ="Zone wise macro properties"


g_point = sns.pointplot(x="Labels",y="Accuracy",data=accuracy_comparison_df,capsize=0.2)
g_point.figure.suptitle(tit1,fontsize=15,fontweight='bold')
g_point.figure.savefig(fig_target_dir+"Accuracy_comparison_"+ipsi_contra+"_"+st_type+".png")


#labels = list(np.unique(sub_dat11_wgp[st_type]))

confusion_mat_final = pd.concat(confusion_mats_all).groupby(level=0).mean()
confusion_mat_final = confusion_mat_final.reindex(index=labels,columns=labels)

confusion_mat_shuff_final = pd.concat(confusion_mats_all_shuff).groupby(level=0).mean()
confusion_mat_shuff_final = confusion_mat_shuff_final.reindex(index=labels,columns=labels)


fig = pl.figure(figsize=(18,8))
subhands = [ fig.add_subplot(121), fig.add_subplot(122)]


sns.heatmap(confusion_mat_final,cmap='magma_r',vmin=0,vmax=0.5,ax=subhands[0],xticklabels=labels,yticklabels=labels,annot=True,square=True)
sns.heatmap(confusion_mat_shuff_final,cmap='magma_r',vmin=0,vmax=0.5,ax=subhands[1],xticklabels=labels,yticklabels=labels,annot=True,square=True)
subhands[0].set_title(tit1+"(Actual:score="+str(np.round(np.mean(score_all),2))+")",fontsize=12,fontweight='bold')
subhands[1].set_title(tit1+"(Shuffled:score="+str(np.round(np.mean(score_shuff_all),2))+")",fontsize=12,fontweight='bold')
for ax in subhands:
    ax.set_xlabel(ax.get_xlabel(),fontsize=15,fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(),fontsize=15,fontweight='bold')


fig.savefig(fig_target_dir+"Confusion_matrix_random_forest_"+ipsi_contra+"_"+st_type+".png")
#fig.savefig(fig_target_dir+"Confusion_matrix_random_forest_"+ipsi_contra+"_"+st_type+".pdf")









