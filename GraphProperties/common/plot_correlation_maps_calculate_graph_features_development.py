"""
Created on Wed Jun 17 14:01:23 2020

Correlation matrix of maps, rearranged correlation matrix 

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
#import analyze as anal
import sys

sys.path.append("common/")
import graph_prop_funcs_analyze as graph_anal

#Raw data here
data_dir = "../SpaethBahugunaData/ProcessedData/Development_Dataset/"

# Store data after preprocessing here
data_target_dir = "data/"
if os.path.isdir("./Figure1/")== False:
	os.mkdir("./Figure1")
fig_target_dir = "./Figure1/"


development = "DEVELOPMENT"
days = os.listdir(data_dir)
data_2d = pickle.load(open(data_target_dir+"data_2d_maps_days.pickle","rb"))
data = pd.read_csv(data_target_dir+"meta_data_days.csv")
cov_2d_dict = deepcopy(data_2d)

# Gamma values for calculating the graph properties 
gammas = np.round(np.arange(0.0,1.5,0.17),2)

graph_properties = dict()


# An example gamma, to display the rearranged correlation maps after louvain community algorithm is run
gamma_re_arrange = 0.34
gamma_re_arrange_ind = np.where(gammas == gamma_re_arrange)[0][0]

# zones
zone_names = ["B_contra","AX_contra","Alat_contra","Amed_contra","Amed_ipsi","Alat_ipsi","AX_ipsi","B_ipsi"]
zone_lims = [(-233,-133),(-133,-108),(-108,-58),(-58,0),(0,50),(50,100),(100,125),(125,235)]

zone_binning = np.arange(-235,235,5)

'''
B_contra : -233 to -133
AX_contra : -133 to -108
Alat_contra : - 108 to -58
Amed_contra : -58 to 0

Amed ipsi :0 to 50
Alat_ipis = 50 to 100
AX_ipsi : 100 to 125
B_ipis : 125 to 285
'''
mat_type = "norm"

#participation_pos_all = []
#participation_neg_all = []
#mdz_all = []

# For different seeds, calculate graph properties
seeds_list = pickle.load(open(data_target_dir+"seeds.pickle","rb"))[:5]
print(seeds_list)
#seed_plot = random.sample(list(seeds_list),1)[0]
seed_plot = int(sys.argv[1])
print(seed_plot)


def store_gp(ci_list_corr,corr_2d,participation_pos_all,participation_neg_all,mdz_all):
	
	zscore = []
	for ci in ci_list_corr:

		zs = graph_anal.calc_module_degree_zscore(corr_2d,ci,True,False)
		zscore.append(zs)
		#rc = anal.calc_rich_club_wu(ci) # Rich club also gave nan
		#rich_club.append(rc)
	# independent of gammas
	mdz_all.append(zscore)
	part_pos, part_neg = graph_anal.calc_participation_coef_sign(corr_2d,ci_list_corr,False,True)
	participation_pos_all.append(part_pos)
	participation_neg_all.append(part_neg)

	# re arranging the correlation matroices
	list_nodes = [ bct.modularity.ci2ls(x1) for x1 in ci_list_corr]
	loc_assort_pos, loc_assort_neg = graph_anal.calc_local_assortativity_sign(corr_2d)

	return zscore,  part_pos, part_neg,  list_nodes, loc_assort_pos, loc_assort_neg






for seed in seeds_list[:5]:
    print(seed)
    # Plot the correlation maps for all animals within a subtype and calculate the graph properties for the subtype  
    participation_pos_all = []
    participation_neg_all = []
    mdz_all = []

    participation_pos_all_null = []
    participation_neg_all_null = []
    mdz_all_null = []

    for st in days:
        data_slice = data.loc[data["days"]==st]
        num_subfigs = len(data_slice)
        fig1 = pl.figure(figsize=(20,20))
        fig2 = pl.figure(figsize=(20,20))
        rows = int(np.round(np.sqrt(num_subfigs)))
        cols = rows
        print(st)
        if rows*cols < num_subfigs:
            rows = rows+1
        subfig_hands1 = []
        subfig_hands2 = []
        
        graph_properties[st] = dict()
        graph_properties[st]["modularity"] = dict()
        graph_properties[st]["indices"] = []
        graph_properties[st]["names"] = []
        fig1.suptitle("Days:"+st,fontsize=15,fontweight='bold') 
        fig2.suptitle("Days:"+st+" rearranged, gamma = "+str(gamma_re_arrange),fontsize=15,fontweight='bold') 
        
        for i,(rn,cn) in enumerate(zip(data_slice["rat_num"],data_slice["cell_num"])):

            subfig_hands1.append(fig1.add_subplot(rows,cols,i+1))
            subfig_hands2.append(fig2.add_subplot(rows,cols,i+1))
            graph_properties[st]["modularity"][i] = dict()
            graph_properties[st]["names"].append(str(rn)+"-"+str(cn))
            if str(cn) in list(data_2d[st][str(rn)].keys()):

                cov_2d = np.cov(data_2d[st][str(rn)][str(cn)]["map"].T)
                tot_amplitude = np.nansum(data_2d[st][str(rn)][str(cn)]["map_nz"])
                avg_amplitude = np.nanmean(data_2d[st][str(rn)][str(cn)]["map_nz"])
                nz_dim = np.shape(data_2d[st][str(rn)][str(cn)]["map"])
                active_sites = (len(np.where(data_2d[st][str(rn)][str(cn)]["map"]  > 3.0)[0])/(nz_dim[0]*nz_dim[1]))*100 # % active sites

                corr_2d = np.corrcoef(data_2d[st][str(rn)][str(cn)]["map"].T,data_2d[st][str(rn)][str(cn)]["map"].T)[:len(cov_2d),:len(cov_2d)]
                ind_nan = np.where(np.isnan(corr_2d)==True)
                if len(ind_nan[0]) > 0:
                    ind_nonan = np.where(np.isnan(corr_2d)==False)
                    xlim = (np.min(np.unique(ind_nonan[0])),np.max(np.unique(ind_nonan[0])))
                    ylim = (np.min(np.unique(ind_nonan[1])),np.max(np.unique(ind_nonan[1])))
                    corr_2d = corr_2d[xlim[0]:xlim[1],ylim[0]:ylim[1]]  # Only consider no nan values of the correlation matrix

                cov_2d_dict[st][str(rn)][str(cn)] = dict()
                cov_2d_dict[st][str(rn)][str(cn)]["cov"] = cov_2d 
                if mat_type == "norm":
                    corr_2d = corr_2d
                cov_2d_dict[st][str(rn)][str(cn)]["corr"] = corr_2d #+np.nanmin(corr_2d)
                
                # Find modularity index
                gammas,num_mods_cov, mod_index_cov,ci_list_cov = graph_anal.calc_modularity(cov_2d)
                _,num_mods_corr, mod_index_corr,ci_list_corr = graph_anal.calc_modularity(corr_2d)
                '''               
                zscore = []
                for ci in ci_list_corr:
                    zs = graph_anal.calc_module_degree_zscore(corr_2d,ci,True,False)
                    zscore.append(zs)

                mdz_all.append(zscore)
                part_pos, part_neg = graph_anal.calc_participation_coef_sign(corr_2d,ci_list_corr,False,True)
                participation_pos_all.append(part_pos)
                participation_neg_all.append(part_neg)

                # re arranging the correlation matroices
                list_nodes = [ bct.modularity.ci2ls(x1) for x1 in ci_list_corr]
           

                re_arranged_corr = graph_anal.get_re_arranged_matrix(ci_list_corr[gamma_re_arrange_ind],corr_2d) 
                loc_assort_pos, loc_assort_neg = graph_anal.calc_local_assortativity_sign(corr_2d)
                '''
                corr_2d_null = bct.randmio_und_signed(corr_2d,5)[0] # This function randomizes an undirected weighted network with positive and negative weights, while simultaneously preserving the degree distribution of positive and negative weights. The function does not preserve the

                cov_2d_dict[st][str(rn)][str(cn)]["corr_null"] = corr_2d_null #+np.nanmin(corr_2d)

                _,num_mods_corr_null, mod_index_corr_null,ci_list_corr_null = graph_anal.calc_modularity(corr_2d_null)

                zscore,  part_pos, part_neg, list_nodes, loc_assort_pos, loc_assort_neg = store_gp(ci_list_corr,corr_2d,participation_pos_all,participation_neg_all,mdz_all)
                zscore_null,  part_pos_null, part_neg_null, list_nodes_null, loc_assort_pos_null, loc_assort_neg_null = store_gp(ci_list_corr_null,corr_2d_null,participation_pos_all_null,participation_neg_all_null,mdz_all_null)

                re_arranged_corr = graph_anal.get_re_arranged_matrix(ci_list_corr[gamma_re_arrange_ind],corr_2d) 
                re_arranged_corr_null = graph_anal.get_re_arranged_matrix(ci_list_corr_null[gamma_re_arrange_ind],corr_2d_null) 


                graph_properties[st]["modularity"][i]["cov"] = (mod_index_cov,num_mods_cov)
                graph_properties[st]["modularity"][i]["corr"] = (mod_index_corr,num_mods_corr)
                graph_properties[st]["modularity"][i]["corr_null"] = (mod_index_corr_null,num_mods_corr_null)
                graph_properties[st]["modularity"][i]["mod_list"] = ci_list_corr
                graph_properties[st]["modularity"][i]["mod_list_null"] = ci_list_corr_null
                graph_properties[st]["modularity"][i]["rearranged_corr"] = re_arranged_corr
                graph_properties[st]["modularity"][i]["rearranged_corr_null"] = re_arranged_corr_null
                graph_properties[st]["modularity"][i]["norm"] = np.linalg.norm(corr_2d) 
                graph_properties[st]["modularity"][i]["total_amplitude"] = np.abs(tot_amplitude)*0.01 # pA
                graph_properties[st]["modularity"][i]["average_amplitude"] = np.abs(avg_amplitude)*0.01 # pA
                graph_properties[st]["modularity"][i]["percentage_active_sites"] = active_sites


                # align node numbers with position in the binned_pos
                if len(ind_nan[0]) > 0:
                    ind_zone_bins = np.digitize(data_2d[st][str(rn)][str(cn)]["pos_centered"][xlim[0]:xlim[1]],bins=zone_binning)
                    graph_properties[st]["modularity"][i]["ind_zone_bins_node_num_mapping"] = (ind_zone_bins,np.arange(xlim[0],xlim[1]))
                else:
                    ind_zone_bins = np.digitize(data_2d[st][str(rn)][str(cn)]["pos_centered"],bins=zone_binning)
                    graph_properties[st]["modularity"][i]["ind_zone_bins_node_num_mapping"] = (ind_zone_bins,np.arange(0,np.shape(data_2d[st][str(rn)][str(cn)]["map"])[1]))

                # Store node wise participation coefficient, module degree zscore and local assortativity coefficient and average it over the zones on the basis of ind_zone_bins 

                graph_properties[st]["modularity"][i]["participation_pos_zone"] = part_pos
                graph_properties[st]["modularity"][i]["participation_neg_zone"] = part_neg
                graph_properties[st]["modularity"][i]["local_assortativity_pos_whole_zone"] = loc_assort_pos

                # Participation coefficient of all nodes in the graph (whole graph participation coefficient is median of these distributions)
                graph_properties[st]["modularity"][i]["participation_whole"] = (np.median(part_pos,axis=1),np.median(part_neg,axis=1))
                graph_properties[st]["modularity"][i]["participation_whole_null"] = (np.median(part_pos_null,axis=1),np.median(part_neg_null,axis=1))


                # Participation coefficient on the hemisphere resolution - ipsilateral and contralateral
                if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                    part_pos_ipsi = np.array(part_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    part_neg_ipsi = np.array(part_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    graph_properties[st]["modularity"][i]["participation_ipsi"] = (np.median(part_pos_ipsi,axis=1),np.median(part_neg_ipsi,axis=1))

                    part_pos_ipsi_null = np.array(part_pos_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    part_neg_ipsi_null = np.array(part_neg_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    graph_properties[st]["modularity"][i]["participation_ipsi_null"] = (np.median(part_pos_ipsi_null,axis=1),np.median(part_neg_ipsi_null,axis=1))


                if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                    part_pos_contra = np.array(part_pos)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    part_neg_contra = np.array(part_neg)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    graph_properties[st]["modularity"][i]["participation_contra"] = (np.median(part_pos_contra,axis=1),np.median(part_neg_contra,axis=1))

                    part_pos_contra_null = np.array(part_pos_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    part_neg_contra_null = np.array(part_neg_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    graph_properties[st]["modularity"][i]["participation_contra_null"] = (np.median(part_pos_contra_null,axis=1),np.median(part_neg_contra_null,axis=1))

			    # local assortativity for all nodes in the graph (whole graph local assortativity is median of these dsitributions)

                graph_properties[st]["modularity"][i]["local_assortativity_whole"] = (np.median(loc_assort_pos))
                graph_properties[st]["modularity"][i]["local_assortativity_whole_null"] = (np.median(loc_assort_pos_null))

                #local assortativity on the hemisphere resolution - ipsilateral and contralateral

                if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                    lim1 = np.min([len(loc_assort_pos),np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                    lim2 = np.min([len(loc_assort_pos),np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                    graph_properties[st]["modularity"][i]["local_assortativity_ipsi"] = (np.median(loc_assort_pos[lim1:lim2]))
                    lim1n = np.min([len(loc_assort_pos_null),np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                    lim2n = np.min([len(loc_assort_pos_null),np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])])
                    graph_properties[st]["modularity"][i]["local_assortativity_ipsi_null"] = (np.median(loc_assort_pos_null[lim1n:lim2n]))


                if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                    lim1 = np.min([len(loc_assort_pos),np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                    lim2 = np.min([len(loc_assort_pos),np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                    graph_properties[st]["modularity"][i]["local_assortativity_contra"] = (np.median(loc_assort_pos[lim1:lim2]))
                    lim1n = np.min([len(loc_assort_pos_null),np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                    lim2n = np.min([len(loc_assort_pos_null),np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])])
                    graph_properties[st]["modularity"][i]["local_assortativity_contra_null"] = (np.median(loc_assort_pos_null[lim1n:lim2n]))

                # Module degree zscore for all nodes in the graph (whole graph module degree zscore is median of these distributions)

                graph_properties[st]["modularity"][i]["module_degree_zscore_whole"] = np.median(zscore,axis=1)
                graph_properties[st]["modularity"][i]["module_degree_zscore_whole_null"] = np.median(zscore_null,axis=1)

                if len(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]) > 0:
                    zscore_ipsi = np.array(zscore)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    graph_properties[st]["modularity"][i]["module_degree_zscore_ipsi"] = np.median(zscore_ipsi,axis=1)
                    zscore_ipsi_null = np.array(zscore_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_ipsi"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_ipsi"])]
                    graph_properties[st]["modularity"][i]["module_degree_zscore_ipsi_null"] = np.median(zscore_ipsi_null,axis=1)

                if len(data_2d[st][str(rn)][str(cn)]["ind_contra"]) > 0:
                    zscore_contra = np.array(zscore)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    graph_properties[st]["modularity"][i]["module_degree_zscore_contra"] = np.median(zscore_contra,axis=1)
                    zscore_contra_null = np.array(zscore_null)[:,np.min(data_2d[st][str(rn)][str(cn)]["ind_contra"]):np.max(data_2d[st][str(rn)][str(cn)]["ind_contra"])]
                    graph_properties[st]["modularity"][i]["module_degree_zscore_contra_null"] = np.median(zscore_contra_null,axis=1)

                  
                graph_properties[st]["indices"].append(i)

                vmin = np.nanmin(cov_2d)/2.
                vmax = np.nanmax(cov_2d)/2.

                vmin = np.nanmin(corr_2d)/2.
                vmax = np.nanmax(corr_2d)/2.
                print(vmin,vmax)
                subfig_hands1[-1].pcolor(corr_2d,cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
                subfig_hands2[-1].pcolor(re_arranged_corr,cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
               
                subfig_hands1[-1].set_aspect('equal')
                subfig_hands2[-1].set_aspect('equal')


            subfig_hands1[-1].set_title("rat num:"+str(rn)+",cell num:"+str(cn),fontsize=12,fontweight='bold')
            subfig_hands2[-1].set_title("rat num:"+str(rn)+",cell num:"+str(cn),fontsize=12,fontweight='bold')
            if i < (num_subfigs-2):
                subfig_hands1[-1].set_xticklabels([])
                subfig_hands2[-1].set_xticklabels([])

        graph_properties[st]["gammas"] = gammas

        if seed == seed_plot: 
            fig1.subplots_adjust(left = 0.05,right=0.96,wspace=0.2,hspace=0.2,bottom=0.06,top=0.95)
            fig2.subplots_adjust(left = 0.05,right=0.96,wspace=0.2,hspace=0.2,bottom=0.06,top=0.95)
            fig1.savefig(fig_target_dir+"corr_maps_"+st+"_"+mat_type+".png")
            fig2.savefig(fig_target_dir+"corr_maps_rearranged_"+st+"_"+mat_type+"_"+str(seed)+".png")



    pickle.dump(cov_2d_dict,open(data_target_dir+"covariance_maps_days_"+mat_type+"_"+str(seed)+".pickle","wb"))
    pickle.dump(graph_properties,open(data_target_dir+"graph_properties_days_"+mat_type+"_"+str(seed)+".pickle","wb"))




