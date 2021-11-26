# CereballarMaps-GraphProperties
Source code for " Cerebellar Connectivity Maps Embody Individual Adaptive Behavior in Mice", Spaeth, Bahuguna et al. 

# Requirements 
Each script was tested and built with an IPython console from Spyder (python 3.6 or later). Anaconda and WinPython distributions were tested in the present study. Please install the following packages in your environnement:
-matplotlib 3.4
-numpy 1.19
-pingouin 0.5
-pandas 1.3
-seaborn 0.11
-statannot 
-scipy 1.6
-statsmodels 0.12
-sklearn 0.24
-neo 0.10
-bctpy 0.5.2
-snakemake
-neo
-openpyxl

# Input Data
Raw and pre-processed synaptic maps are available below:
Ludovic Spaeth, Jyotika Bahuguna, Demian Battaglia, & Philippe Isope. (2021). GC-PC_Cerebellar_Connectivity _Maps [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5714670

Download and unzip SpaethBahugunaData.zip - set this file as DataDir in scripts from Synaptic_Maps_and_Behavior

DataSource : Source_Data_Spaeth_Bahuguna_et_al.xlsx

# Scripts
Scripts are sorted according to the output panels from each figures in the paper.
Please specify the path to the dataset mentionned at the top of each script (i.e. the location where you saved the COMPLETE_DATASET folder on your machine). Present assumption is current directory.
Please specify an output path on your machine at the top of each script (i.e. where to save the plots and sheets generated by the script). 
Build the project by calling snakemake. This may take several minutes..

```bash
snakemake
```
