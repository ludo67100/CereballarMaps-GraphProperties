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
import seaborn as sns
import scipy.spatial.distance as sp_sp_dist
from itertools import product
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
import sklearn as skl
from sklearn.cluster import KMeans
from matplotlib.gridspec import GridSpec
import scipy.stats as sp_st
from matplotlib.markers import TICKDOWN

import graph_prop_funcs_analyze as graph_anal

gammas = np.arange(0.0,1.5,0.17)





def mean_over_gammas(field_names,orig_df,temp_dict,fields_avged_gamma,data_type='subtype'):
    for x in orig_df.groupby(field_names):
        for k in fields_avged_gamma:
            temp_tuple = (x[1][k].mean(),np.unique(x[1][data_type])[0],x[0])
            temp_dict[k].append(tuple(np.hstack(temp_tuple))) # field, subtype,name
    # Replace "short_names" by "names"
    if isinstance(field_names,list):
        field_names_new = [ "names" if x == "short_names" else x for x in field_names]
    elif isinstance(field_names,str):
        field_names_new = [field_names]
    for i,k in enumerate(fields_avged_gamma):
        temp_df = pd.DataFrame(temp_dict[k],columns=[k,data_type]+field_names_new)

        if i == 0:
            merged = temp_df
        else:
            merged = merged.merge(temp_df,on=[data_type]+field_names_new)


    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna()

    return merged


def median_over_gammas(field_names,orig_df,temp_dict,fields_avged_gamma,data_type='subtype'):
    for x in orig_df.groupby(field_names):
        for k in fields_avged_gamma:
            temp_tuple = (x[1][k].median(),np.unique(x[1][data_type])[0],x[0])
            temp_dict[k].append(tuple(np.hstack(temp_tuple))) # field, subtype,name
    # Replace "short_names" by "names"
    if isinstance(field_names,list):
        field_names_new = [ "names" if x == "short_names" else x for x in field_names]
    elif isinstance(field_names,str):
        field_names_new = [field_names]
    for i,k in enumerate(fields_avged_gamma):
        temp_df = pd.DataFrame(temp_dict[k],columns=[k,data_type]+field_names_new)

        if i == 0:
            merged = temp_df
        else:
            merged = merged.merge(temp_df,on=[data_type]+field_names_new)


    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna()

    return merged





def significance_bar(start, end, height, displaystring,ax, linewidth=1.2, markersize=8, boxpad=0.3, fontsize=15,color='k'):
    # draw a line with downticks at the ends
    ax.plot([start, end], [height]*2, '-', color=color, lw=linewidth,
             marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    # draw the text with a bounding box covering up the line
    ax.text(0.5*(start+end), height, displaystring, ha='center', va='center', bbox=dict(
        facecolor='1.', edgecolor='none', boxstyle='Square,pad='+str(boxpad)), size=fontsize)


def plot_scatter_plot_errorbars(data,props,color,marker,ax,ms=10,alpha=1.0,use_median=True,return_pts=False):
    #x_mean = np.nanmean(data[props[0]])
    if use_median == True:
        x_mean = np.nanmedian(data[props[0]])
    else:
        x_mean = np.nanmean(data[props[0]])
    #y_mean = np.nanmedian(z[1]['module_degree_zscore'])
    #y_mean = np.nanmean(data[props[1]])
    if use_median == True:
        y_mean = np.nanmedian(data[props[1]])
    else:
        y_mean = np.nanmean(data[props[1]])
    if use_median == True:
        x_err = list(np.abs(x_mean-np.array([np.percentile(data[props[0]],25), np.percentile(data[props[0]],75)])))

        y_err = list(np.abs(y_mean-np.array([np.percentile(data[props[1]],25), np.percentile(data[props[1]],75)])))
    else:
        #x_err = np.nanstd(data[props[0]])#/len(data[props[0]])
        x_err = mean_confidence_interval(data[props[0]])#/len(data[props[0]])
        #y_err = np.nanstd(data[props[1]])#/len(data[props[1]])
        y_err = mean_confidence_interval(data[props[1]])#/len(data[props[1]])
    #print("x_err",x_err)

    print("x_mean",x_mean)
    print("y_mean",y_mean)
    #print("y_err",y_err)
    ax.errorbar(x=[x_mean],y=[y_mean],xerr=np.array(list([x_err])).T,yerr=np.array(list([y_err])).T,capsize=0.1,ls='',color=color,fmt=marker,markersize=ms,capthick=2,alpha=alpha,lw=3.5)
    if return_pts == True:
        return x_mean,y_mean

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp_st.sem(a)
    h = se * sp_st.t.ppf((1 + confidence) / 2., n-1)
    return h

def mean_confidence_interval_all(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sp_st.sem(a)
    h = se * sp_st.t.ppf((1 + confidence) / 2., n-1)
    return m,h,m-h,m+h




def boxplot_bold_labels(g,g_prop,vline_prop,hline_prop):
    #for ax in g.axes:                             
    ax = g.axes                                    
    ax.set_xlabel("",fontsize=25,fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(),fontsize=20,fontweight='bold')
    if g_prop["title"] != "":
        tit = ax.get_title().split('=')[1]         
        ax.set_title(tit,fontsize=15,fontweight='bold') 
         
    for x in ax.get_xticklabels():
        x.set_fontsize(15)
        x.set_fontweight('bold')                   
        #x.set_visible(False)                      
        
    for x in ax.get_yticklabels():
        x.set_fontsize(15)                         
        x.set_fontweight('bold')                   
    
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    if vline_prop["flag"] == "y":                  
        ax.vlines(x=vline_prop["x"],ymin=ylims[0],ymax=ylims[1],linestyles='dashed',linewidth=vline_prop["lw"],color=vline_prop["color"])                 
        
    if hline_prop["flag"] == "y":                  
        ax.hlines(y=hline_prop["y"],xmin=xlims[0],xmax=xlims[1],linestyles='dashed',linewidth=hline_prop["lw"],color=hline_prop["color"])                 
    
    
        


def convert_map_graph(map_orig,gamma_re_arrange_ind):

    cov_2d = np.cov(map_orig.T)
    corr_2d = np.corrcoef(map_orig.T,map_orig.T)[:len(cov_2d),:len(cov_2d)]
    ind_nan = np.where(np.isnan(corr_2d)==True)
    if len(ind_nan[0]) > 0:
        ind_nonan = np.where(np.isnan(corr_2d)==False)
        xlim = (np.min(np.unique(ind_nonan[0])),np.max(np.unique(ind_nonan[0])))
        ylim = (np.min(np.unique(ind_nonan[1])),np.max(np.unique(ind_nonan[1])))
        corr_2d = corr_2d[xlim[0]:xlim[1],ylim[0]:ylim[1]]

    gammas,num_mods_corr, mod_index_corr,ci_list_corr = graph_anal.calc_modularity(corr_2d)
    re_arranged_corr = graph_anal.get_re_arranged_matrix(ci_list_corr[gamma_re_arrange_ind],corr_2d)

    return re_arranged_corr,num_mods_corr,mod_index_corr,ci_list_corr,corr_2d

