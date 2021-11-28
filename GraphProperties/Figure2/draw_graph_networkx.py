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
import networkx as nx
import matplotlib as mpl

sys.path.append("./common/")
import analyze as anal
data_dir = "./data/"
data_target_dir = "./data/"
fig_target_dir = "./Figure2/"


seeds = list(pickle.load(open(data_target_dir+"seeds.pickle","rb")))[:5] # For full simulation run for all seeds

#seed_plot = random.sample(list(SEEDS),1)[0]
seed_plot = str(seeds[0])
data_2d = pickle.load(open(data_target_dir+"data_2d_maps_days.pickle","rb"))

dat_type = sys.argv[1]
st = sys.argv[2] # days or subtypes
#rn = sys.argv[3] # rat number
#cn = sys.argv[4] # cell number
gammas = np.round(np.arange(0.0,1.5,0.17),2)

if dat_type == "subtypes":
    graph_prop = pickle.load(open(data_dir+"graph_properties_norm_"+seed_plot+".pickle","rb"))
    data = pd.read_csv(data_target_dir+"meta_data.csv")
elif dat_type == "days":
    graph_prop = pickle.load(open(data_dir+"graph_properties_days_norm_"+seed_plot+".pickle","rb"))
    data = pd.read_csv(data_target_dir+"meta_data_days.csv")

data_slice = data.loc[data[dat_type]==st]
colors_mods = ["darkorange","steelblue",'darkolivegreen', 'saddlebrown']

mod_index = 6 # corresponding to modularity index ~ 1.02
#mod_index = 5 # corresponding to modularity index ~ 1.02
#seed = 234
seed = np.random.randint(0,9999999)
modularity_indices = []
mdz_all = []
participation_coef = []
names = []

for i,(rn1,cn1) in enumerate(zip(data_slice["rat_num"],data_slice["cell_num"])): 

    names.append((rn1,cn1))
    modularity_indices.append(graph_prop[st]["modularity"][i]["corr"][0][6])
    mdz_all.append(graph_prop[st]["modularity"][i]['module_degree_zscore_whole'][6])
    participation_coef.append(graph_prop[st]["modularity"][i]['participation_whole'][0][6])



if st == "P9P10":
    
    rn = str(150410)
    cn = str(1)
    seed = 9979345
elif st == "P14P18":
    rn = str(151217) # Taking a bit less extreme example, hence not the index 0 but 1
    cn = str(3)
    seed = 4535458
elif st == "P30P40":
    rn = str(160503)
    cn = str(1)
    seed = 8945445
elif st == "P12P13":
    rn = str(150407)
    cn = str(1)
    seed = 9703288

print(seed)
np.random.seed(seed)

print(rn,cn)
for i,(rn1,cn1) in enumerate(zip(data_slice["rat_num"],data_slice["cell_num"])):
    
    if str(rn1) == str(rn) and str(cn1) == str(cn):
        map_orig = data_2d[st][rn][cn]['map']
        rearranged_corr,num_mods_corr,mod_index_corr,ci_list_corr,corr_2d = anal.convert_map_graph(map_orig,6)
        mod_ind_list = num_mods_corr[mod_index]
        print("modularity_index",mod_index_corr[mod_index])
        mod_ind_val = np.round(mod_index_corr[mod_index],2)
        rearranged_corr[rearranged_corr<0] = 0
        g_full = nx.DiGraph(rearranged_corr)
        for n in g_full.nodes():
            ind_lie = np.digitize(n,bins=np.cumsum(mod_ind_list))
            g_full.nodes[n]["mod_num"] = ind_lie

            

        fig = pl.figure(figsize=(16,16))
        t1 = fig.add_subplot(111)
        thresh = 0.0
        pos = nx.spring_layout(g_full,iterations=200,k=0.1)
        col_list = [mpl.colors.to_rgba(colors_mods[g_full.nodes[x]["mod_num"]]) for x in g_full.nodes()]

        nx.draw_networkx_nodes(g_full,node_size=600,node_color=col_list, alpha=1.0,arrows=True,arrowsize=10,ax=t1,pos=pos)
        for edge in g_full.edges(data="weight"):
            nx.draw_networkx_edges(g_full,pos=pos,edgelist=[edge], edge_color='grey',width=edge[2]*5,arrows=True,arrowsize=10,ax=t1,arrowstyle='fancy',connectionstyle= 'arc3,rad=0.2',alpha=0.1)
        t1.set_title(st+":"+"rat num:"+str(rn)+", cell num:"+str(cn)+" (modularity index:"+str(mod_ind_val)+")",fontsize=15,fontweight='bold')
        fig.savefig(fig_target_dir+"Graph_"+st+".png")
    
