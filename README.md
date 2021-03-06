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

# Scripts - Synaptic Maps and Behavior
Scripts are sorted according to the output panels from each figures in the paper. Tu run a script, open it in any python-based script editor (e.g. Spyder). Please specify the path to the dataset mentionned at the top of each script (i.e. the location where you saved the SpaethBahugunaData folder and DataSource file on your machine): 

#----------------Adjust dataSource and saveDir path------------------------

#Input folder (SpaethBahugunaData) or file (Source_Data)

dataDir = 'path/SpaethBahugunaData'

OR

file = 'path/Source_Data_Spaeth_Bahuguna_et_al.xlsx'

#Savedir 

saveDir =  'where to save plots and sheets on your machine'

#--------------------------The code----------------------------------------

You should not modify parameters below in order to reproduce the panels shown in the paper. 

Each scripts performs calculations and displays panels from Figure 2a-b-d, 3a-c-d, 4a-b-c-e-f, 5a-b-c-f.  

Random Forest in Figure 5b was built with an Orange Data Mining (ODM) workflow. ODM can be installed via Anaconda Navigator or from here: https://orangedatamining.com/
Simply open tSNE_RF_Synaptic_Params_zonewise_Segregated_Training_5B.ows in ODM and set Avg_Amplitude_Active_Sites_STR-LTR-segregated.xlsx as input data. 




# Scripts - GraphProperties
Download and unzip SpaethBahugunaData.zipi in the current folder. The folder "SpaethBahugunaData/" serves as the main data directory
 
The intermediate data is stored in "GraphProperties/data/". The whole project can be built by calling "snakemake". 

GraphProperties/Snakefile - makefile that lists all dependencies and rules to generate the figures.   

Build the project by calling snakemake. This may take several minutes..

```bash
snakemake
```
