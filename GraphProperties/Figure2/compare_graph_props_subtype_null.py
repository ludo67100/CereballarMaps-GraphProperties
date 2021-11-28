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
import bct
from collections import Counter 
import matplotlib.cm as cm
import sys
import seaborn as sns
import scipy.stats as sp_stats

sys.path.append("./common/")
import analyze as anal
data_dir = "./data/"
data_target_dir = "./data/"
fig_target_dir = "./Figure2/"


Fig2_panel_name = dict({"modularity_index":"H","participation_pos":"I","module_degree_zscore":"J","local_assortativity_pos_whole":"K"})



subtype = sys.argv[1]
ipsi_contra = sys.argv[2]

if subtype == "subtype":
	if ipsi_contra == "n":
		graph_prop_df = pd.read_csv(data_dir+"graph_properties_pandas_all.csv")
		graph_prop_df_null = pd.read_csv(data_dir+"graph_properties_pandas_null_all.csv")
		
		prop_names = ["data_type","modularity_index","participation_pos","module_degree_zscore","local_assortativity_pos_whole","gamma","names"]+[subtype]
		non_string_columns = [ "modularity_index","participation_pos","module_degree_zscore","local_assortativity_pos_whole"]

	elif ipsi_contra == "y":
		graph_prop_df = pd.read_csv(data_dir+"graph_properties_pandas_sub_contra_ipsi_all.csv")
		graph_prop_df_null = pd.read_csv(data_dir+"graph_properties_pandas_sub_contra_ipsi_all_null.csv")
		prop_names = ["data_type","modularity_index","participation_pos_ipsi","participation_pos_contra","module_degree_zscore_ipsi","module_degree_zscore_contra","local_assortativity_pos_ipsi","local_assortativity_pos_contra","gamma","names"]+[subtype]
		non_string_columns = [ "modularity_index","participation_pos_ipsi","participation_pos_contra","module_degree_zscore_ipsi","module_degree_zscore_contra","local_assortativity_pos_ipsi","local_assortativity_pos_contra"]

	#elif ipsi_contra == "y":
	 #	graph_properties_with_behavior_pandas_sub_ipsi_contra_all	
elif subtype == "development":
	graph_prop_df = pd.read_csv(data_dir+"graph_properties_pandas_days_all.csv")
	graph_prop_df_null = pd.read_csv(data_dir+"graph_properties_pandas_days_null_all.csv")
	prop_names = ["data_type","modularity_index","participation_pos","module_degree_zscore","local_assortativity_pos_whole","gamma","names"]+[subtype]
	non_string_columns = [ "modularity_index","participation_pos","module_degree_zscore","local_assortativity_pos_whole"]

graph_prop_df["data_type"] = "Actual"

graph_prop_df_null["data_type"] = "NULL"

 
def avg_over_gammas_and_animals(field,data,non_string_columns,metric='mean'):                        
    temp_dat1_wgp = dict()
    for k in non_string_columns:                                                                     
        temp_dat1_wgp[k] = []
        
    #sub_dat1_wgp = anal.mean_over_gammas(field,data,temp_dat1_wgp,non_string_columns)               
    if metric == "mean":
        sub_dat1_wgp = anal.mean_over_gammas(field,data,temp_dat1_wgp,non_string_columns,"data_type")            
    elif metric == "median":
        sub_dat1_wgp = anal.median_over_gammas(field,data,temp_dat1_wgp,non_string_columns,"data_type")          
    for k in non_string_columns:
        sub_dat1_wgp[k] = sub_dat1_wgp[k].astype('float')                                            
        
    return sub_dat1_wgp         

string_cols = list(set(prop_names)-set(non_string_columns))

sub_data = avg_over_gammas_and_animals(["names"]+[subtype],graph_prop_df[prop_names],non_string_columns)
sub_data_null = avg_over_gammas_and_animals(["names"]+[subtype],graph_prop_df_null[prop_names],non_string_columns)
sub_data_rel = pd.DataFrame(columns=sub_data.columns)

'''
for grp1,grp2 in zip(sub_data.groupby("names"),sub_data_null.groupby("names")):
	gp1 = grp1[1][non_string_columns]
	gp2 = grp2[1][non_string_columns]
	gp_rel = (gp1-gp2)/gp2
	for sc in string_cols:
		if sc == "gamma":
			continue
		gp_rel[sc] = grp1[1][sc]

	sub_data_rel = sub_data_rel.append(gp_rel)
'''
for ns in non_string_columns:
	med = sub_data_null[ns].median()
	sub_data_rel[ns] = (sub_data[ns] - med)/med

for sc in string_cols:
	if sc == "gamma":
		continue
	sub_data_rel[sc] = sub_data[sc]

sub_data_rel.to_csv(data_target_dir+"Graph_properties_relative_to_null_graphs_"+subtype+"_"+ipsi_contra+".csv")
sub_data.to_csv(data_target_dir+"Graph_properties_graphs_"+subtype+"_"+ipsi_contra+".csv")
sub_data_null.to_csv(data_target_dir+"Graph_properties_graphs_null_"+subtype+"_"+ipsi_contra+".csv")




subtypes = np.unique(sub_data[subtype])


if subtype == "development":
	subtype_ord = ["P9P10","P12P13","P14P18","P30P40"]

	rel_colors = ["skyblue","cadetblue","dodgerblue","mediumblue"]
else:
	subtype_ord = subtypes
	rel_colors = ["skyblue","forestgreen","salmon","gray","orange","purple"]


for st in subtypes:
	sub_data_st = sub_data.loc[sub_data[subtype]==st]	
	sub_data_st_null = sub_data_null.loc[sub_data_null[subtype]==st]	

	sub_data_st = sub_data_st.drop(columns=[subtype])
	sub_data_st_null = sub_data_st_null.drop(columns=[subtype])

	sub_data_rel_st = sub_data_rel.loc[sub_data_rel[subtype]==st]
	sub_data_rel_st = sub_data_rel_st.drop(columns=[subtype])

	sub_data_rearranged = sub_data_st.melt(['names',"data_type"])
	sub_data_null_rearranged = sub_data_st_null.melt(['names',"data_type"])
	sub_data_rel_rearranged = sub_data_rel_st.melt(['names',"data_type"])

	final_data = sub_data_rearranged.append(sub_data_null_rearranged)

	fig = pl.figure(figsize=(16,16))
	t1 = fig.add_subplot(111)

	g1 = sns.pointplot(x="variable",y="value",hue="data_type",data=final_data,palette='magma_r',linewidth=4.5,ax=t1,scale=2.0)	

	positions = g1.axes.get_xticks()
	pos_labels = [ x.get_text() for x in g1.axes.get_xticklabels()] 

	for i,x1 in enumerate(non_string_columns):
		dat1 = np.array(sub_data_st[x1])
		dat2 = np.array(sub_data_st_null[x1])
		t_stat,p_val = sp_stats.mannwhitneyu(dat1,dat2) # Does not assume equal variance

		print(t_stat,p_val)
		if p_val/2. < 0.05 :
			if p_val/2. < 0.01:
				# and  sub_samp_mean_pow_trials[j] > 1.5*sub_samp_mean_pow_trials[i]: #and significance[i][j][0] < 0:
				if p_val < 0.0001:
					displaystring = r'***'
				elif p_val < 0.001:
					displaystring = r'**'
				else:
					displaystring = r'*'

				max_dat = np.max(np.hstack((dat1,dat2)))
				y_sig = max_dat+(i)*0.1*max_dat
				ind1 = np.where(np.array(pos_labels)==x1)[0][0]
				ind2 = np.where(np.array(pos_labels)==x1)[0][0]
					#if x1 == "WT" :
				anal.significance_bar(positions[ind1]-0.2,positions[ind1]+0.2,y_sig,displaystring,g1.axes)


	g1.figure.suptitle(st,fontsize=20,fontweight='bold')
	g1.axes.set_ylabel(g1.axes.get_ylabel(),fontsize=15,fontweight='bold')
	for x in g1.axes.get_xticklabels():
		x.set_fontsize(15)
		x.set_fontweight('bold')
		x.set_rotation(45)

	g1.axes.set_xlabel("")
	g1.axes.set_ylabel(g1.axes.get_ylabel(),fontsize=20,fontweight='bold')
	g1.figure.subplots_adjust(bottom=0.2)
	g1.legend(fontsize=15)
	g1.figure.savefig(fig_target_dir+"Comparison_graph_props_"+st+"_"+subtype+"_"+ipsi_contra+"_null.png")


for ns in non_string_columns:
	pl.figure()	
	#	pdb.set_trace()
	g2 = sns.boxplot(x=subtype,y=ns,hue=subtype,data=sub_data_rel,palette=rel_colors,linewidth=4.5,order=subtype_ord,dodge=False,showfliers=True)
	#g2 = sns.boxplot(x=subtype,y=ns,hue=subtype,data=sub_data_rel,palette=rel_colors,linewidth=4.5,order=subtype_ord,dodge=False,showfliers=False)
	'''	
	if ns == "modularity_index":
		# Because log(0) is not defined, you will have to plot the line before log y-axis

		xlims = g2.axes.get_xlim()
		g2.axes.hlines(y=0,xmin=xlims[0],xmax=xlims[1],linestyles='dashed',linewidth=2.5,color='k')
		g2.axes.set_yscale('log')
		flag = "n"
	else:
	'''
	flag = "y"	
	anal.boxplot_bold_labels(g2,dict({"title":""}), dict({"flag":"n"}),dict({"flag":flag,"y":0,"lw":2.5,"color":"k"}))
	g2.figure.suptitle(ns,fontsize=25,fontweight='bold')	
	g2.figure.subplots_adjust(top=0.85,left=0.1)
	g2.figure.savefig(fig_target_dir+"Figure2_"+Fig2_panel_name[ns]+"_"+subtype+"_"+ipsi_contra+".png")
