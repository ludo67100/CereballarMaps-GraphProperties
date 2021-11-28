"""
Created on Wed Jun 17 14:01:23 2020

combine graph properties for different seeds

@author: Jyotika.bahuguna
"""




import os
import glob
import numpy as np
import pylab as pl
import scipy.io as sio
from copy import copy, deepcopy
import pickle
import matplotlib.cm as cm
import pdb
import h5py
import pandas as pd
import bct
from collections import Counter 
import matplotlib.cm as cm
import analyze as anal
import sys


#  
data_target_dir = "data/"


num_or_size = "size" # num of clusters or size of the largest cluster
data_type = sys.argv[1]


gamma_re_arrange = 0.34

if data_type == "subtype":

    electrophys = "ELECTROPHY"
    # Raw data
    data_dir = "../SpaethBahugunaData/ProcessedData/Adaptive_Dataset/"

    subtypes = os.listdir(data_dir)
    #data_2d = pickle.load(open(data_target_dir+"data_2d_maps.pickle","rb"))
    #data = pd.read_csv(data_target_dir+"meta_data.csv")

    files = glob.glob(data_target_dir+"graph_properties_norm_*.pickle")

elif data_type == "development":

    development = "DEVELOPMENT"
    # Raw data
    data_dir = "../SpaethBahugunaData/ProcessedData/Development_Dataset/"

    subtypes = os.listdir(data_dir) # Just the name of the variable is subtypes, its actually days
    #data_2d = pickle.load(open(data_target_dir+"data_2d_maps_days.pickle","rb"))
    #data = pd.read_csv(data_target_dir+"meta_data_days.csv")


    files = glob.glob(data_target_dir+"graph_properties_days_norm_*.pickle")


gammas = np.arange(0.0,1.5,0.17)

graph_prop_simps = dict()
graph_prop_simps_null = dict()

percentile = 70
dat_type = data_type

for f in files:
    seed = f.split('/')[-1].split('_')[-1].split('.')[0]
    graph_properties = pickle.load(open(f,"rb"))

    graph_prop_df = pd.DataFrame(columns=["modularity_index","gamma","norms","participation_pos_whole","participation_pos_ipsi","participation_pos_contra","participation_neg_whole","participation_neg_ipsi","participation_neg_contra","local_assortativity_pos_whole","local_assortativity_pos_ipsi","local_assortativity_pos_contra","module_degree_zscore_whole","module_degree_zscore_ipsi","module_degree_zscore_contra","names"]+[dat_type])
    graph_prop_df_null = pd.DataFrame(columns=["modularity_index","gamma","norms","participation_pos_whole","participation_pos_ipsi","participation_pos_contra","participation_neg_whole","participation_neg_ipsi","participation_neg_contra","local_assortativity_pos_whole","local_assortativity_pos_ipsi","local_assortativity_pos_contra","module_degree_zscore_whole","module_degree_zscore_ipsi","module_degree_zscore_contra","names"]+[dat_type])

    temp_dict = dict()
    for x in list(graph_prop_df.keys()):
        temp_dict[x] = []

    temp_dict_null = dict()
    for x in list(graph_prop_df_null.keys()):
        temp_dict_null[x] = []


    for i,st in enumerate(subtypes):
        st_list_cov=[]
        st_mods_list_cov=[]
        st_list_corr=[]
        st_list_corr_null=[]
        st_mods_list_corr=[]
        st_mods_list_corr_null=[]
        norms =[]
        graph_prop_simps[st] = dict()
        graph_prop_simps_null[st] = dict()
        
        participation_pos_whole = []
        participation_pos_whole_null = []
        participation_pos_ipsi = []
        participation_pos_ipsi_null = []
        participation_pos_contra = []
        participation_pos_contra_null = []
        participation_neg_whole = []
        participation_neg_whole_null = []
        participation_neg_ipsi = []
        participation_neg_ipsi_null = []
        participation_neg_contra = []
        participation_neg_contra_null = []
        loc_ass_pos_whole = []
        loc_ass_pos_whole_null = []
        loc_ass_pos_ipsi = []
        loc_ass_pos_ipsi_null = []
        loc_ass_pos_contra = []
        loc_ass_pos_contra_null = []
        zscore_whole = []
        zscore_whole_null = []
        zscore_ipsi = []
        zscore_ipsi_null = []
        zscore_contra = []
        zscore_contra_null = []
        names=[]
        nz_inds = []
        count = 0
        print("==================================================================")
        print(st)
        print("==================================================================")
        for j,x in enumerate(list(graph_properties[st]["modularity"].keys())):
            ind = graph_properties[st]["indices"]
            for y1 in list(graph_properties[st]["modularity"][x].keys()):
                print(y1)
                if "norm" in y1:        
                    norms.append(graph_properties[st]["modularity"][x]["norm"])
                elif "participation" in y1 and "whole" in y1:
                    if "null" in y1:
                        participation_pos_whole_null.append(graph_properties[st]["modularity"][x]["participation_whole_null"][0])
                        participation_neg_whole_null.append(graph_properties[st]["modularity"][x]["participation_whole_null"][1])

                    else:
                        participation_pos_whole.append(graph_properties[st]["modularity"][x]["participation_whole"][0])
                        participation_neg_whole.append(graph_properties[st]["modularity"][x]["participation_whole"][1])
                elif "participation" in y1 and "ipsi" in y1:
                    if "null" in y1:
                        participation_pos_ipsi_null.append(graph_properties[st]["modularity"][x]["participation_ipsi_null"][0])
                        participation_neg_ipsi_null.append(graph_properties[st]["modularity"][x]["participation_ipsi_null"][1])

                    else:
                        participation_pos_ipsi.append(graph_properties[st]["modularity"][x]["participation_ipsi"][0])
                        participation_neg_ipsi.append(graph_properties[st]["modularity"][x]["participation_ipsi"][1])
                elif "participation" in y1 and "contra" in y1:
                    if "null" in y1:
                        participation_pos_contra_null.append(graph_properties[st]["modularity"][x]["participation_contra_null"][0])
                        participation_neg_contra_null.append(graph_properties[st]["modularity"][x]["participation_contra_null"][1])

                    else:
                        participation_pos_contra.append(graph_properties[st]["modularity"][x]["participation_contra"][0])
                        participation_neg_contra.append(graph_properties[st]["modularity"][x]["participation_contra"][1])
                elif "zscore" in y1 and "whole" in y1:
                    if "null" in y1:
                        zscore_whole_null.append(graph_properties[st]["modularity"][x]["module_degree_zscore_whole_null"])
                    else:
                        zscore_whole.append(graph_properties[st]["modularity"][x]["module_degree_zscore_whole"])
                elif "zscore" in y1 and "ipsi" in y1:
                    if "null" in y1:
                        zscore_ipsi_null.append(graph_properties[st]["modularity"][x]["module_degree_zscore_ipsi_null"])
                    else:
                        zscore_ipsi.append(graph_properties[st]["modularity"][x]["module_degree_zscore_ipsi"])
                elif "zscore" in y1 and "contra" in y1:
                    if "null" in y1:
                        zscore_contra_null.append(graph_properties[st]["modularity"][x]["module_degree_zscore_contra_null"])
                    else:
                        zscore_contra.append(graph_properties[st]["modularity"][x]["module_degree_zscore_contra"])
              
                elif "local" in y1:
                    if "null" in y1:
                        loc_ass_pos_whole_null.append(graph_properties[st]["modularity"][x]["local_assortativity_whole_null"])
                        loc_ass_pos_ipsi_null.append(graph_properties[st]["modularity"][x]["local_assortativity_ipsi_null"])
                        loc_ass_pos_contra_null.append(graph_properties[st]["modularity"][x]["local_assortativity_contra_null"])
                             
                    else:
                        loc_ass_pos_whole.append(graph_properties[st]["modularity"][x]["local_assortativity_whole"])
                        loc_ass_pos_ipsi.append(graph_properties[st]["modularity"][x]["local_assortativity_ipsi"])
                        loc_ass_pos_contra.append(graph_properties[st]["modularity"][x]["local_assortativity_contra"])
                elif y1 == "cov" or y1 == "corr":
                    mod_indices = graph_properties[st]["modularity"][x][y1][0]
                    num_mods = [len(y) for y in  graph_properties[st]["modularity"][x][y1][1]]
                    if num_mods[0] == 0:
                        continue
                    num_mods_size = [np.max(y) for y in  graph_properties[st]["modularity"][x][y1][1] if len(y) > 0] 
                    
                    num_mods_greater_size = [ len(np.where(np.array(y) >= np.percentile(y,percentile))[0])  for y in  graph_properties[st]["modularity"][x][y1][1] if len(y) > 0]
                    nz_inds.append(x)
            
                    print(mod_indices)
                    print(num_mods)
                    if "cov" in y1:
                        st_list_cov.append((mod_indices,num_mods,num_mods_size,num_mods_greater_size))
                        st_mods_list_cov.append(graph_properties[st]["modularity"][x][y1][1])
                    elif "corr" in y1:
                        st_list_corr.append((mod_indices,num_mods,num_mods_size,num_mods_greater_size))
                        st_mods_list_corr.append(graph_properties[st]["modularity"][x][y1][1])

                elif y1 == "corr_null":
                    mod_indices_null = graph_properties[st]["modularity"][x][y1][0]
                    #if num_or_size == "num":
                    num_mods_null = [len(y) for y in  graph_properties[st]["modularity"][x][y1][1]]
                    # If num_mods are zero just go to next data point, because if this empty, causes problems, while slicing by gammas
                    if num_mods_null[0] == 0:
                        continue
                    #elif num_or_size == "size":
                    num_mods_size_null = [np.max(y) for y in  graph_properties[st]["modularity"][x][y1][1] if len(y) > 0] 
                    
                    num_mods_greater_size_null = [ len(np.where(np.array(y) >= np.percentile(y,percentile))[0])  for y in  graph_properties[st]["modularity"][x][y1][1] if len(y) > 0]
                    st_list_corr_null.append((mod_indices_null,num_mods_null,num_mods_size_null,num_mods_greater_size_null))
                    st_mods_list_corr_null.append(graph_properties[st]["modularity"][x][y1][1])

        print("participation_pos_ipsi_null",participation_pos_ipsi_null)

        graph_prop_simps_null[st]["st_list_corr_null"] = st_list_corr_null
        graph_prop_simps_null[st]["st_mods_list_corr_null"] = st_mods_list_corr_null

        graph_prop_simps[st]["participation_pos_whole"] = participation_pos_whole
        graph_prop_simps_null[st]["participation_pos_whole_null"] = participation_pos_whole_null
        graph_prop_simps[st]["participation_pos_ipsi"] = participation_pos_ipsi
        graph_prop_simps_null[st]["participation_pos_ipsi_null"] = participation_pos_ipsi_null
        graph_prop_simps[st]["participation_pos_contra"] = participation_pos_contra
        graph_prop_simps_null[st]["participation_pos_contra_null"] = participation_pos_contra_null
        graph_prop_simps[st]["participation_neg_whole"] = participation_neg_whole
        graph_prop_simps_null[st]["participation_neg_whole_null"] = participation_neg_whole_null
        graph_prop_simps[st]["participation_neg_ipsi"] = participation_neg_ipsi
        graph_prop_simps_null[st]["participation_neg_ipsi_null"] = participation_neg_ipsi_null
        graph_prop_simps[st]["participation_neg_contra"] = participation_neg_contra
        graph_prop_simps_null[st]["participation_neg_contra_null"] = participation_neg_contra_null
        graph_prop_simps[st]["module_degree_zscore_whole"] = zscore_whole
        graph_prop_simps_null[st]["module_degree_zscore_whole_null"] = zscore_whole_null
        graph_prop_simps[st]["module_degree_zscore_ipsi"] = zscore_ipsi
        graph_prop_simps_null[st]["module_degree_zscore_ipsi_null"] = zscore_ipsi_null
        graph_prop_simps[st]["module_degree_zscore_contra"] = zscore_contra
        graph_prop_simps_null[st]["module_degree_zscore_contra_null"] = zscore_contra_null

        print(len(norms),len(st_list_corr))

        nz_inds = np.unique(nz_inds)
        if len(loc_ass_pos_whole) > len(st_list_corr):
            graph_prop_simps[st]["local_assortativity_pos_whole"] = np.array(loc_ass_pos_whole)[nz_inds]
            graph_prop_simps_null[st]["local_assortativity_pos_whole_null"] = np.array(loc_ass_pos_whole_null)[nz_inds]
            graph_prop_simps[st]["local_assortativity_pos_ipsi"] = np.array(loc_ass_pos_ipsi)[nz_inds]
            graph_prop_simps_null[st]["local_assortativity_pos_ipsi_null"] = np.array(loc_ass_pos_ipsi_null)[nz_inds]
            graph_prop_simps[st]["local_assortativity_pos_contra"] = np.array(loc_ass_pos_contra)[nz_inds]
            graph_prop_simps_null[st]["local_assortativity_pos_contra_null"] = np.array(loc_ass_pos_contra_null)[nz_inds]
        else:
            graph_prop_simps[st]["local_assortativity_pos_whole"] = np.array(loc_ass_pos_whole)
            graph_prop_simps_null[st]["local_assortativity_pos_whole_null"] = np.array(loc_ass_pos_whole_null)
            graph_prop_simps[st]["local_assortativity_pos_ipsi"] = np.array(loc_ass_pos_ipsi)
            graph_prop_simps_null[st]["local_assortativity_pos_ipsi_null"] = np.array(loc_ass_pos_ipsi_null)
            graph_prop_simps[st]["local_assortativity_pos_contra"] = np.array(loc_ass_pos_contra)
            graph_prop_simps_null[st]["local_assortativity_pos_contra_null"] = np.array(loc_ass_pos_contra_null)

        if len(graph_properties[st]['names']) > len(st_list_corr):
            graph_prop_simps[st]["names"] = np.array(graph_properties[st]['names'])[nz_inds]
            graph_prop_simps_null[st]["names"] = np.array(graph_properties[st]['names'])[nz_inds]
        else:
            graph_prop_simps[st]["names"] = np.array(graph_properties[st]['names'])
            graph_prop_simps_null[st]["names"] = np.array(graph_properties[st]['names'])

        if num_or_size == "num":
            ind_prop = 1
        elif num_or_size == "size":
            ind_prop = 2
        for k in np.arange(0,len(gammas)):
            
            temp_dict["modularity_index"].append(np.array(st_list_corr)[:,:,k][:,0])
            temp_dict_null["modularity_index"].append(np.array(st_list_corr_null)[:,:,k][:,0])

            nz_inds = np.unique(nz_inds)
            temp_dict["gamma"].append([ np.round(gammas[k],2) for i2 in np.arange(0,len(np.array(st_list_corr)[:,:,k][:,0]))])

            temp_dict_null["gamma"].append([ np.round(gammas[k],2) for i2 in np.arange(0,len(np.array(st_list_corr_null)[:,:,k][:,0]))])


            if len(norms) > len(st_list_corr):
                temp_dict["norms"].append(np.array(norms)[nz_inds])
                temp_dict["participation_pos_whole"].append(np.array(graph_prop_simps[st]["participation_pos_whole"])[nz_inds,k])
                temp_dict_null["participation_pos_whole"].append(np.array(graph_prop_simps_null[st]["participation_pos_whole_null"])[nz_inds,k])
                temp_dict["participation_pos_ipsi"].append(np.array(graph_prop_simps[st]["participation_pos_ipsi"])[nz_inds,k])
                temp_dict_null["participation_pos_ipsi"].append(np.array(graph_prop_simps_null[st]["participation_pos_ipsi_null"])[nz_inds,k])
                temp_dict["participation_pos_contra"].append(np.array(graph_prop_simps[st]["participation_pos_contra"])[nz_inds,k])

                temp_dict_null["participation_pos_contra"].append(np.array(graph_prop_simps_null[st]["participation_pos_contra_null"])[nz_inds,k])
                temp_dict["participation_neg_whole"].append(np.array(graph_prop_simps[st]["participation_neg_whole"])[nz_inds,k])
                temp_dict["participation_neg_ipsi"].append(np.array(graph_prop_simps[st]["participation_neg_ipsi"])[nz_inds,k])
                temp_dict["participation_neg_contra"].append(np.array(graph_prop_simps[st]["participation_neg_contra"])[nz_inds,k])
                temp_dict["module_degree_zscore_whole"].append(np.array(graph_prop_simps[st]["module_degree_zscore_whole"])[nz_inds,k])
                temp_dict_null["module_degree_zscore_whole"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_whole_null"])[nz_inds,k])
                temp_dict["module_degree_zscore_ipsi"].append(np.array(graph_prop_simps[st]["module_degree_zscore_ipsi"])[nz_inds,k])

                temp_dict_null["module_degree_zscore_ipsi"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_ipsi_null"])[nz_inds,k])
                temp_dict["module_degree_zscore_contra"].append(np.array(graph_prop_simps[st]["module_degree_zscore_contra"])[nz_inds,k])

                temp_dict_null["module_degree_zscore_contra"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_contra_null"])[nz_inds,k])
            else:
                temp_dict["norms"].append(np.array(norms))
                temp_dict["participation_pos_whole"].append(np.array(graph_prop_simps[st]["participation_pos_whole"])[:,k])
                temp_dict_null["participation_pos_whole"].append(np.array(graph_prop_simps_null[st]["participation_pos_whole_null"])[:,k])

                temp_dict["participation_pos_ipsi"].append(np.array(graph_prop_simps[st]["participation_pos_ipsi"])[:,k])
                print(graph_prop_simps_null[st]["participation_pos_ipsi_null"])
                temp_dict_null["participation_pos_ipsi"].append(np.array(graph_prop_simps_null[st]["participation_pos_ipsi_null"])[:,k])

                temp_dict["participation_pos_contra"].append(np.array(graph_prop_simps[st]["participation_pos_contra"])[:,k])
                temp_dict_null["participation_pos_contra"].append(np.array(graph_prop_simps_null[st]["participation_pos_contra_null"])[:,k])
                temp_dict["participation_neg_whole"].append(np.array(graph_prop_simps[st]["participation_neg_whole"])[:,k])
                temp_dict["participation_neg_ipsi"].append(np.array(graph_prop_simps[st]["participation_neg_ipsi"])[:,k])
                temp_dict["participation_neg_contra"].append(np.array(graph_prop_simps[st]["participation_neg_contra"])[:,k])
                temp_dict["module_degree_zscore_whole"].append(np.array(graph_prop_simps[st]["module_degree_zscore_whole"])[:,k])
                temp_dict_null["module_degree_zscore_whole"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_whole_null"])[:,k])


                temp_dict["module_degree_zscore_ipsi"].append(np.array(graph_prop_simps[st]["module_degree_zscore_ipsi"])[:,k])
                temp_dict_null["module_degree_zscore_ipsi"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_ipsi_null"])[:,k])

                temp_dict["module_degree_zscore_contra"].append(np.array(graph_prop_simps[st]["module_degree_zscore_contra"])[:,k])
                temp_dict_null["module_degree_zscore_contra"].append(np.array(graph_prop_simps_null[st]["module_degree_zscore_contra_null"])[:,k])

            if len(names) > len(st_list_corr):
                temp_dict["names"].append(np.array(graph_prop_simps[st]["names"])[nz_inds])
                temp_dict_null["names"].append(np.array(graph_prop_simps_null[st]["names"])[nz_inds])
            else:
                temp_dict["names"].append(np.array(graph_prop_simps[st]["names"]))
                temp_dict_null["names"].append(np.array(graph_prop_simps_null[st]["names"]))
            temp_dict["local_assortativity_pos_whole"].append(np.array(graph_prop_simps[st]["local_assortativity_pos_whole"]))
            
            temp_dict_null["local_assortativity_pos_whole"].append(np.array(graph_prop_simps_null[st]["local_assortativity_pos_whole_null"]))


            temp_dict["local_assortativity_pos_ipsi"].append(np.array(graph_prop_simps[st]["local_assortativity_pos_ipsi"]))
            temp_dict_null["local_assortativity_pos_ipsi"].append(np.array(graph_prop_simps_null[st]["local_assortativity_pos_ipsi_null"]))

            temp_dict["local_assortativity_pos_contra"].append(np.array(graph_prop_simps[st]["local_assortativity_pos_contra"]))
            temp_dict_null["local_assortativity_pos_contra"].append(np.array(graph_prop_simps_null[st]["local_assortativity_pos_contra_null"]))

            count+=len(np.array(st_list_corr)[:,:,k][:,0]) 
            
        temp_dict[dat_type].append( [st for i3 in np.arange(0,count)])

        temp_dict_null[dat_type].append( [st for i3 in np.arange(0,count)])
        print(st)
        print(len(st_list_cov))
        print(len(st_list_corr))




    graph_prop_df["modularity_index"] = np.hstack(temp_dict["modularity_index"])
    graph_prop_df_null["modularity_index"] = np.hstack(temp_dict_null["modularity_index"])

    graph_prop_df["norms"] = np.hstack(temp_dict["norms"])
    graph_prop_df["participation_pos_whole"] = np.hstack(temp_dict["participation_pos_whole"])
    graph_prop_df_null["participation_pos_whole"] = np.hstack(temp_dict_null["participation_pos_whole"])
    graph_prop_df["participation_pos_ipsi"] = np.hstack(temp_dict["participation_pos_ipsi"])
    graph_prop_df_null["participation_pos_ipsi"] = np.hstack(temp_dict_null["participation_pos_ipsi"])

    graph_prop_df["participation_pos_contra"] = np.hstack(temp_dict["participation_pos_contra"])
    graph_prop_df_null["participation_pos_contra"] = np.hstack(temp_dict_null["participation_pos_contra"])
    graph_prop_df["local_assortativity_pos_whole"] = np.hstack(temp_dict["local_assortativity_pos_whole"])
    graph_prop_df_null["local_assortativity_pos_whole"] = np.hstack(temp_dict_null["local_assortativity_pos_whole"])

    graph_prop_df["local_assortativity_pos_ipsi"] = np.hstack(temp_dict["local_assortativity_pos_ipsi"])
    graph_prop_df_null["local_assortativity_pos_ipsi"] = np.hstack(temp_dict_null["local_assortativity_pos_ipsi"])


    graph_prop_df["local_assortativity_pos_contra"] = np.hstack(temp_dict["local_assortativity_pos_contra"])
    graph_prop_df_null["local_assortativity_pos_contra"] = np.hstack(temp_dict_null["local_assortativity_pos_contra"])
    graph_prop_df["participation_neg_whole"] = np.hstack(temp_dict["participation_neg_whole"])
    graph_prop_df["participation_neg_ipsi"] = np.hstack(temp_dict["participation_neg_ipsi"])
    graph_prop_df["participation_neg_contra"] = np.hstack(temp_dict["participation_neg_contra"])
    graph_prop_df["module_degree_zscore_whole"] = np.hstack(temp_dict["module_degree_zscore_whole"])
    graph_prop_df_null["module_degree_zscore_whole"] = np.hstack(temp_dict_null["module_degree_zscore_whole"])


    graph_prop_df["module_degree_zscore_ipsi"] = np.hstack(temp_dict["module_degree_zscore_ipsi"])
    graph_prop_df_null["module_degree_zscore_ipsi"] = np.hstack(temp_dict_null["module_degree_zscore_ipsi"])
    graph_prop_df["module_degree_zscore_contra"] = np.hstack(temp_dict["module_degree_zscore_contra"])
    graph_prop_df_null["module_degree_zscore_contra"] = np.hstack(temp_dict_null["module_degree_zscore_contra"])
    graph_prop_df["names"] = np.hstack(temp_dict["names"])
    graph_prop_df_null["names"] = np.hstack(temp_dict_null["names"])


    graph_prop_df["gamma"] = np.hstack(temp_dict["gamma"])
    graph_prop_df_null["gamma"] = np.hstack(temp_dict_null["gamma"])
    #graph_prop_df["subtype"] = np.hstack(temp_dict["subtype"])
    graph_prop_df[dat_type] = np.hstack(temp_dict[dat_type])
    graph_prop_df_null[dat_type] = np.hstack(temp_dict_null[dat_type])


    graph_prop_df = graph_prop_df.replace([np.inf, -np.inf], np.nan)

    if data_type == "subtype":
        graph_prop_df.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_"+seed+".csv")
        graph_prop_df_null.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_null_"+seed+".csv")
    elif data_type == "development":
        graph_prop_df.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_days_"+seed+".csv")
    graph_prop_df_nonan = graph_prop_df.dropna(axis=0)

    graph_prop_df_nonan_null = graph_prop_df_null.dropna(axis=0)

    if data_type == "subtype":
        graph_prop_df_nonan.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_"+seed+".csv")

        graph_prop_df_nonan_null.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_null_"+seed+".csv")
    elif data_type == "development":
        graph_prop_df_nonan.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_days_"+seed+".csv")

if data_type == "subtype":
    files1 = glob.glob(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_[0-9]*.csv")
    files2 = glob.glob(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_[0-9]*.csv")
    files3 = glob.glob(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_null_[0-9]*.csv")
    files3_null = glob.glob(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_null_[0-9]*.csv")

elif data_type == "development":
    files1 = glob.glob(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_days_[0-9]*.csv")
    files2 = glob.glob(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_days_[0-9]*.csv")

def merge_df_seeds(files):

    for i,f in enumerate(files):
        temp_df = pd.read_csv(f)
        seed = f.split('/')[-1].split('_')[-1].split('.')[0]
        temp_df["seed"] = seed

        if i == 0:
            merge_df = temp_df
        else:
            merge_df = merge_df.append(temp_df)
    return merge_df

if data_type == "subtype":
    merge_df1 = merge_df_seeds(files1)
    merge_df1.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_all.csv")

    merge_df2 = merge_df_seeds(files2)
    merge_df2.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_all.csv")

    merge_df1_null = merge_df_seeds(files3)
    merge_df1_null.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_all_null.csv")

    merge_df2_null = merge_df_seeds(files3_null)
    merge_df2_null.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_all_null.csv")

elif data_type == "development":
    merge_df1 = merge_df_seeds(files1)
    merge_df1.to_csv(data_target_dir+"graph_properties_pandas_for_behav_sub_contra_ipsi_days_all.csv")

    merge_df2 = merge_df_seeds(files2)
    merge_df2.to_csv(data_target_dir+"graph_properties_pandas_sub_contra_ipsi_days_all.csv")



