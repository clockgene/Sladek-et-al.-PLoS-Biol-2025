"""
Created on Tue Feb 28 13:01:23 2023
v.20250905
@author: martin.sladek
"""

import seaborn as sns
import pandas as pd
import numpy as np
import os, re
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.multivariate.pca import PCA
from tkinter import filedialog
from tkinter import *
from matplotlib_venn import venn2
import math
from sklearn.preprocessing import StandardScaler

##################### Functions ########################################################################################################
# Standardize data - apply the z-score method in Pandas using the .mean() and .std() methods
def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()        
    return df_std    
    # call the z_score function
    # df_cars_standardized = z_score(df_cars)

  
# PCA - principle component analysis, unsupervised dimensionality reduction technique, finds direction of maximum variation 
def PCA_plot(data, columns, hue, name, mydir, pca_model, pc='PC2'):

    ## To save as vector svg with fonts editable in Corel ###
    import matplotlib as mpl                                                                       #import matplotlib as mpl
    new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    mpl.rcParams.update(new_rc_params)
    
    # scree plot to show PC scores
    pca_model.plot_scree(log_scale=False)
    plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'PCA_scree_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    
    # scatter plot of PC1 x PC2
    data['PC1'] = pca_model.loadings.iloc[:, 0] # pca_model.loadings['comp_0'] >> or 'comp_01'
    data['PC2'] = pca_model.loadings.iloc[:, 1] # pca_model.loadings['comp_1']
    fig = plt.figure(figsize=(5, 5))
    sns.scatterplot(x="PC1", y="PC2", data=data, hue=hue, s=30)
    plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'PCA_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()

    # histogram of selected PC by hue
    sns.histplot(data=data, x=pc, hue=hue, kde=True, stat="density") 
    ###### Calculate p values between hue for separate categories in pc ######
    by_hue = data.groupby(data[hue])  # for ANOVA and labels, auto-create col_order
    hue_order = []
    for a, frame in by_hue:
        hue_order.append(a) 
    ##### ANOVA ######
    alist = []
    for i in range(len(hue_order)):
        alist.append(data[pc][(data[hue] == hue_order[i])].dropna(how='any'))
    F, p = stats.f_oneway(*alist) # asterisk is for *args - common idiom to allow arbitrary number of arguments to functions    
    if p < 0.0000000001:
        plt.annotate('ANOVA p < 1e-10', xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                    textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    else:
        plt.annotate('ANOVA p = ' + str(round(p, 8)), xy=(1, 1), xycoords='axes fraction', fontsize=10, xytext=(-15, -15), 
                    textcoords='offset points', horizontalalignment='right', verticalalignment='top')    
    plt.savefig(f'{mydir}' + '\\' + f'PCA_histo_{hue}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'PCA_histo_{hue}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()
    
    # description in txt file
    description = open(f'{mydir}\\PCA_{hue}_{name}.txt', 'w')
    description.write(f'{columns} \nANOVA F = {F}, p = {p}')
    description.close()   
    
    return data['PC1'], data['PC2']


# Benjamini-Hochberg FDR
def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

# ttest/MW followed by FDR
# FDR_compare(data, data.columns[3:], hue = 'regime', sub = 'sex', subgroup = 'F')
def FDR_compare(data, cols, hue, sub=False, subgroup=False, test='ttest'):
    # import statsmodels
    by_hue = data.groupby(data[hue])  # for ANOVA and labels, auto-create col_order
    hue_order = []
    for a, frame in by_hue:
        hue_order.append(a)   
    
    if test == 'ttest':
        
        if sub is False:
            
            datax1 = data[cols][(data[hue] == hue_order[0])].dropna(how='any')
            datax2 = data[cols][(data[hue] == hue_order[1])].dropna(how='any')
            
            _, p = stats.ttest_ind(datax1, datax2)
            
            # _, p, _ = statsmodels.stats.weightstats.ttest_ind(datax1, datax2)
            
        else:
            
            datax1 = data[cols][(data[hue] == hue_order[0]) & (data[sub] == subgroup)]
            datax2 = data[cols][(data[hue] == hue_order[1]) & (data[sub] == subgroup)]   
            
            _, p = stats.ttest_ind(datax1, datax2)
            
            # _, p, _ = statsmodels.stats.weightstats.ttest_ind(datax1, datax2)
        
    if test == 'mannwhitneyu':
        
        if sub is False:
            
            datax1 = data[cols][(data[hue] == hue_order[0])].dropna(how='any')
            datax2 = data[cols][(data[hue] == hue_order[1])].dropna(how='any')
            
            _, p = stats.mannwhitneyu(datax1, datax2)           
            
            
        else:
            
            datax1 = data[cols][(data[hue] == hue_order[0]) & (data[sub] == subgroup)]
            datax2 = data[cols][(data[hue] == hue_order[1]) & (data[sub] == subgroup)]
            
            _, p = stats.mannwhitneyu(datax1, datax2)    
    
    p_adj = fdr(p)
    ##alternative
    # from statsmodels.stats.multitest import fdrcorrection
    # _, p_adj = fdrcorrection(p)    
    
    df = pd.DataFrame(p_adj, index =  cols, columns = ['p_adjusted'])
    
    return df  



# Define a function to split each value in the Timepoint column
def split_timepoint(timepoint):
    pattern = r'([a-zA-Z])(\d+(\.\d+)?)'
    match = re.match(pattern, timepoint)
    if match:
        return pd.Series([match.group(1), match.group(2)], index=['Letter', 'Digits'])
    else:
        return pd.Series([None, None], index=['Letter', 'Digits'])
    


def Venn(left, right, middle, labels, mydir):  # venn2(subsets = (980, 1197, 679), set_labels = ('Group A', 'Group B'))
    sns.set_context("paper", font_scale=1.4) 
    venn2(subsets = (left, right, middle), set_labels = labels)
    plt.savefig(f'{mydir}' + '\\' + f'Venn {labels}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'Venn {labels}.svg', format = 'svg', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()

       
#  function to bar plot values with filtering by combination of p values
def value_count_plots_filter_sortcounts(data, col, pval_f1, mydir, pval_f2=np.nan, alpha1=0.05, alpha2=0.05):
    sns.set_context("paper", font_scale=0.9) 
    df1 = data.loc[(data[pval_f1[0]] < alpha1), col]
    df2 = data.loc[(data[pval_f1[0]] < alpha1) & (data[pval_f1[1]] < alpha1), col]
    df3 = data.loc[(data[pval_f1[1]] < alpha1), col]
    
    if pval_f2 == pval_f2:
        df1 = data.loc[(data[pval_f1[0]] < alpha1) & (data[pval_f2[0]] < alpha2), col]
        df3 = data.loc[(data[pval_f1[1]] < alpha1) & (data[pval_f2[1]] < alpha2), col]
        df2 = data.loc[(data[pval_f1[0]] < alpha1) & (data[pval_f1[1]] < alpha1) & (data[pval_f2[0]] < alpha2) & (data[pval_f2[1]] < alpha2), col]                       
    
    fig, ax = plt.subplots(1, 3, sharey=True)  # figsize=(8, 8)
    g1 = sns.barplot(x='index', y=col, data=df1.value_counts(ascending=True).to_frame().reset_index(), ax=ax[0])
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
    g1.set(title=f'{pval_f1[0][0]}_group')
    try:          
        g2 = sns.barplot(x='index', y=col, data=df2.value_counts(ascending=True).to_frame().reset_index(), ax=ax[1])
        g2.set(xlabel=None)
        g2.set(ylabel=None)
        g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
        g2.set(title=f'{pval_f1[0][0]}{pval_f1[1][0]}_group')
    except ValueError:
        pass        
    g3 = sns.barplot(x='index', y=col, data=df3.value_counts(ascending=True).to_frame().reset_index(), ax=ax[2])
    g3.set(xlabel=None)
    g3.set(ylabel=None)
    g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
    g3.set(title=f'{pval_f1[1][0]}_group')

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f'{mydir}\\_{pval_f1[0]}_group.svg', format = 'svg', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\_{pval_f1[0]}_group.png', format = 'png', bbox_inches = 'tight') 
    plt.show()
    plt.clf()
    plt.close()  


# list all metabolites of a selected Cluster
def value_count_plots_cluster(data, cluster, col, mydir, combined=False):
    # plt.figure(figsize=(6,3))
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    sns.set_context("poster", font_scale=1.1) 
    df1 = data.loc[(data['Cluster'] == cluster), col]
    if combined == True:
        df1 = data.loc[(data['Cluster'] == cluster) | (data['Cluster2'] == cluster), 'Class']
    sns.barplot(x='index', y=col, data=df1.value_counts().to_frame().reset_index().sort_values(by=['index']))
    plt.xticks(rotation=90)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    plt.xlabel('Class')
    plt.ylabel('Counts')
    plt.title=f'Cluster {cluster}'
    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f'{mydir}\\Cluster {cluster} barplot.svg', format = 'svg', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\Cluster {cluster} barplot.png', format = 'png', bbox_inches = 'tight') 
    plt.show()
    plt.clf()
    plt.close()    


# Compare 2 histograms with ttest, differentiated by hue (categorical column)
def TT_Histograms1(dataC, dataX, mydir, bins = 'auto', stat = "density", test = 'mw', name='raw'):     # stat="frequency", stat="count" 
    suptitle_all = f'{x_lab} by {stat}'
    
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].hist(dataC, bins=bins)  # bins = 'auto'
    ax[0].hist(dataC, bins=bins, color='slateblue')
    ax[0].set_title('C', fontsize=12, fontweight='bold')
    ax[1].hist(dataX, bins=bins)
    ax[1].hist(dataX, bins=bins, color='tomato')
    ax[1].set_title('S', fontsize=12, fontweight='bold')
    plt.suptitle('Log Amplitude', fontsize=14, fontweight='bold')        

    ###### Calculate t test p values between hue_dat for separate categories in col_dat ######
    pvalues = []
    datax1 = dataC.dropna(how='any')
    datax2 = dataX.dropna(how='any')
    
    if test == 'ttest':
        t, p = stats.ttest_ind(datax1.values, datax2.values)
        pvalues = pvalues + [p]
        plt.annotate('t test \nP = ' + str(round(p, 6)), xy=(1, 1), xycoords='axes fraction', fontsize=10, 
                     xytext=(-5, -5), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    else:
        t, p = stats.mannwhitneyu(datax1.values, datax2.values)    
        pvalues = pvalues + [p]
        plt.annotate('Mann-Whitney \nP = ' + str(round(p, 6)), xy=(1, 1), xycoords='axes fraction', fontsize=10, 
                     xytext=(-5, -5), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    
    plt.savefig(f'{mydir}' + '\\' + f'TT_Histograms1 {suptitle_all}_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'TT_Histograms1 {suptitle_all}_{name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.clf()
    plt.close()


# plots traces of multiple metabolites from all analyzed tissues/ages
def plot_name_ages(data, name, y_norm=True, errorbar='sem', log=True):    # errorbar='std'    
    df=pd.DataFrame()
    
    if log == True:
        df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
    else:
        df = data.loc[:, 'score_E19_ZT00_1':'score_P28_ZT24_5']    # norm to all ages and all tissues
    
    # df.insert(0, 'Name', data['Metabolite name'])  
    df.insert(0, 'Name', data['Name']) 
    dfn = df.loc[df['Name'] == name, df.columns[1:]]  
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
    # Split the Timepoint column into two separate columns for time and replicate 
    # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
    data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
    data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
    # Convert Time and Replicate columns to numeric types
    data_melted["Time"] = pd.to_numeric(data_melted["Time"])
    data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])
    ax = plt.subplot(111)       

    P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
    # Group the data by 'time' and calculate the mean and standard deviation
    grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    # Extract the values for plotting
    time_points = grouped_P28.index
    mean_values_P28 = grouped_P28['Value', 'mean']
    std_values_P28 = grouped_P28['Value', f'{errorbar}']            
    # Plot the line plot with error bars
    line1 = ax.errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4)

    E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
    grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_E19 = grouped_E19['Value', 'mean']
    std_values_E19 = grouped_E19['Value', f'{errorbar}']      
    line2 = ax.errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
    
    P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
    grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P02 = grouped_P02['Value', 'mean']
    std_values_P02 = grouped_P02['Value', f'{errorbar}']      
    line3 = ax.errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)    

    P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
    grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P10 = grouped_P10['Value', 'mean']
    std_values_P10 = grouped_P10['Value', f'{errorbar}']      
    line4 = ax.errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)       

    P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
    grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P20 = grouped_P20['Value', 'mean']
    std_values_P20 = grouped_P20['Value', f'{errorbar}']      
    line5 = ax.errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
    
    if y_norm == True:
        if log == True:
            ymin = math.floor(data_melted['Value'].min())
            ax.set_ylim(ymin)
        if log == False:                
            ymax = math.ceil(data_melted['Value'].max())
            ymin = math.floor(data_melted['Value'].min())
            ax.set_ylim(ymin, ymax)

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    # axs[i, j].set_yticks([])
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f'{name}')  # pad does not work on bottom plots, unless y=1.001 workaround  
    ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 

    # Show the plot
    plt.savefig(f'{mydir}' + '\\' + f'Trace ages {name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'Trace ages {name}.svg', format = 'svg', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()


# plots traces of multiple metabolites from selected tisues only
def plot_name_ages_select_tissue(data, name, tissue='SCN', y_norm=True, errorbar='sem', log=True, annotation=False):    # errorbar='std', tissue='LIV' or 'PLS' or 'SCN'
    df=pd.DataFrame()
    
    if log == True:
        if tissue == 'SCN':
            df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
        if tissue == 'PLS':
            df = data.loc[:, 'log2_PLS_E19_ZT00_1':'log2_PLS_P28_ZT24_5']      
        if tissue == 'LIV':
            df = data.loc[:, 'log2_LIV_E19_ZT00_1':'log2_LIV_P28_ZT24_5']  
    else:
        if tissue == 'SCN':
            df = data.loc[:, 'raw_E19_ZT00_1':'raw_P28_ZT24_5']
        if tissue == 'PLS':
            df = data.loc[:, 'raw_PLS_E19_ZT00_1':'raw_PLS_P28_ZT24_5']      
        if tissue == 'LIV':
            df = data.loc[:, 'raw_LIV_E19_ZT00_1':'raw_LIV_P28_ZT24_5']  
    
    # df.insert(0, 'Name', data['Metabolite name'])  
    df.insert(0, 'Name', data['Name']) 
    dfn = df.loc[df['Name'] == name, df.columns[1:]]  
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
    # Split the Timepoint column into two separate columns for time and replicate 
    # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
    data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
    data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
    # Convert Time and Replicate columns to numeric types
    data_melted["Time"] = pd.to_numeric(data_melted["Time"])
    data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])
    ax = plt.subplot(111)       

    P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
    # Group the data by 'time' and calculate the mean and standard deviation
    grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    # Extract the values for plotting
    time_points = grouped_P28.index
    mean_values_P28 = grouped_P28['Value', 'mean']
    std_values_P28 = grouped_P28['Value', f'{errorbar}']            
    # Plot the line plot with error bars
    line1 = ax.errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4)

    E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
    grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_E19 = grouped_E19['Value', 'mean']
    std_values_E19 = grouped_E19['Value', f'{errorbar}']      
    line2 = ax.errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
    
    P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
    grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P02 = grouped_P02['Value', 'mean']
    std_values_P02 = grouped_P02['Value', f'{errorbar}']      
    line3 = ax.errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)    

    P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
    grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P10 = grouped_P10['Value', 'mean']
    std_values_P10 = grouped_P10['Value', f'{errorbar}']      
    line4 = ax.errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)       

    P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
    grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
    mean_values_P20 = grouped_P20['Value', 'mean']
    std_values_P20 = grouped_P20['Value', f'{errorbar}']      
    line5 = ax.errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
    
    if y_norm == True:
        if log == True:
            ymin = math.floor(data_melted['Value'].min())
            ax.set_ylim(ymin)
        if log == False:                
            ymax = math.ceil(data_melted['Value'].max())
            ymin = math.floor(data_melted['Value'].min())
            ax.set_ylim(ymin, ymax)

    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    # axs[i, j].set_yticks([])
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(f'{name}')  # pad does not work on bottom plots, unless y=1.001 workaround  
    ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
    
    pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4))
    pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))
    
    if annotation == True: # f'empirical p value \n for C = {pC}\n for X = {pX}\n BioC Q value \n for C = {pbC}\n for X = {pbX}'
        ax.annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                      xytext=(-5, -100), textcoords='offset points', horizontalalignment='left', verticalalignment='top', fontsize=18)

    # Show the plot
    plt.savefig(f'{mydir}' + '\\' + f'Trace ages {name} tissue {tissue}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}' + '\\' + f'Trace ages {name} tissue {tissue}.svg', format = 'svg', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()
    

# as above but included p values and other tweaks
def plot_gene_group(data, name_list, title, annotation=True, describe=False, des_list=[], y_norm=True, errorbar='sem', fnt=80, ano2w=True, style='plain', pad=-2):  # errorbar='std' style='sci'
    
    ### To save as vector svg with fonts editable in Corel ###
    # mpl.use('svg')                                                                          #import matplotlib as mpl
    # new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    # mpl.rcParams.update(new_rc_params)

    counter = 0
    Nr = len(name_list)    
    Nc = math.sqrt(Nr) #find the square root of x
    Nw = int(Nr/Nc)
    if Nc.is_integer() == False: #check y is not a whole number
        Nw = math.ceil(Nc)
        Nc = int(round(Nc, 0))        
    Nc = int(Nc)
    figsize = (Nc*3, Nw*3)
    
    fig, axs = plt.subplots(Nw, Nc, sharex=True, figsize=figsize)
    for i in range(Nw):
        for j in range(Nc):                
            
            if Nw*Nc > Nr:
                if counter == Nr:
                    break  
                            
            df=pd.DataFrame()
            # df['Name'] = data['Name']            
            df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']      
            df.insert(0, 'Name', data['Name']) 
            
            name = name_list[counter]
            # Select columns with the specified data label
            dfn = df.loc[df['Name'] == name, df.columns[1:]]               
            
            # Reshape the data into long format using melt
            data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
            # Split the Timepoint column into two separate columns for time and replicate 
            # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
            data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
            data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
            # Convert Time and Replicate columns to numeric types
            data_melted["Time"] = pd.to_numeric(data_melted["Time"])
            data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])      

            P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
            # Group the data by 'time' and calculate the mean and standard deviation
            grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            # Extract the values for plotting
            time_points = grouped_P28.index
            mean_values_P28 = grouped_P28['Value', 'mean']
            std_values_P28 = grouped_P28['Value', f'{errorbar}']            
            # Plot the line plot with error bars
            line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4)

            E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
            grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_E19 = grouped_E19['Value', 'mean']
            std_values_E19 = grouped_E19['Value', f'{errorbar}']      
            line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
            
            P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
            grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P02 = grouped_P02['Value', 'mean']
            std_values_P02 = grouped_P02['Value', f'{errorbar}']      
            line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)    

            P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
            grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P10 = grouped_P10['Value', 'mean']
            std_values_P10 = grouped_P10['Value', f'{errorbar}']      
            line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)       

            P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
            grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P20 = grouped_P20['Value', 'mean']
            std_values_P20 = grouped_P20['Value', f'{errorbar}']      
            line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
            
            if y_norm == True:
                ymin = math.floor(data_melted['Value'].min())
                axs[i, j].set_ylim(ymin)
               
            # if ano2w == True:
            #     model = ols('Value ~ Time + Group + Time:Group', data=data_melted).fit()
            #     result = sm.stats.anova_lm(model, type=2)
            #     p2wano = result['PR(>F)']['Group']                

    
            # # Add a title and axis labels            
            axs[i, j].ticklabel_format(axis='y', style=style, scilimits=(0,0), useMathText=True, useOffset=True) # force scientific notation of all y labels      
            axs[i, j].tick_params(axis='y', which='major', pad=pad, width=1) # move lables closer to y ticks                        
            axs[i, j].set_xticks([0, 4, 8, 12, 16, 20, 24])
            axs[i, j].tick_params(axis='x', which='major', pad=0, width=1)
            # axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_linewidth(1)
            axs[i, j].spines['left'].set_linewidth(1)    
            if describe == True:
                des_name = des_list[counter]
                axs[i, j].set_title(f'{name} {des_name}', pad=-5, y=1.001)
            else:
                axs[i, j].set_title(f'{name}',  pad=-5, y=1.001)  # pad does not work on bottom plots, unless y=1.001 workaround
            
            # needs work
            # ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
            [[line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28']]
            
            # pvalues
            # p2A = round(p2wano, 4)
            pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4))
            pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))
            
            if annotation == True: # f'empirical p value \n for C = {pC}\n for X = {pX}\n BioC Q value \n for C = {pbC}\n for X = {pbX}'
                axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                              xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
        
            counter += 1
           
    plt.suptitle(f"{title}") 
    plt.savefig(f'Traces_{title}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'Traces_{title}.svg', format = 'svg', bbox_inches = 'tight')  
    plt.show()
    plt.clf()
    plt.close()   



# version for a specific tissue only
def plot_gene_group_select_tissue(data, name_list, title, tissue='SCN', annotation=True, describe=False, des_list=[], y_norm=True, errorbar='sem', fnt=80, ano2w=True, style='plain', pad=-2):  # errorbar='std' style='sci'
    
    ### To save as vector svg with fonts editable in Corel ###
    # mpl.use('svg')                                                                          #import matplotlib as mpl
    # new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    # mpl.rcParams.update(new_rc_params)

    counter = 0
    Nr = len(name_list)    
    Nc = math.sqrt(Nr) #find the square root of x
    Nw = int(Nr/Nc)
    if Nc.is_integer() == False: #check y is not a whole number
        Nw = math.ceil(Nc)
        Nc = int(round(Nc, 0))        
    Nc = int(Nc)
    figsize = (Nc*3, Nw*3)
    
    fig, axs = plt.subplots(Nw, Nc, sharex=True, figsize=figsize)
    for i in range(Nw):
        for j in range(Nc):                
            
            if Nw*Nc > Nr:
                if counter == Nr:
                    break  
                            
            df=pd.DataFrame()
            # df['Name'] = data['Name']            
            # df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']  
            if tissue == 'SCN':
                df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
            if tissue == 'PLS':
                df = data.loc[:, 'log2_PLS_E19_ZT00_1':'log2_PLS_P28_ZT24_5']      
            if tissue == 'LIV':
                df = data.loc[:, 'log2_LIV_E19_ZT00_1':'log2_LIV_P28_ZT24_5']  

            df.insert(0, 'Name', data['Name'])             
            name = name_list[counter]
            # Select columns with the specified data label
            dfn = df.loc[df['Name'] == name, df.columns[1:]]               
            
            # Reshape the data into long format using melt
            data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
            # Split the Timepoint column into two separate columns for time and replicate 
            # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
            data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
            data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
            # Convert Time and Replicate columns to numeric types
            data_melted["Time"] = pd.to_numeric(data_melted["Time"])
            data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])      

            P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
            # Group the data by 'time' and calculate the mean and standard deviation
            grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            # Extract the values for plotting
            time_points = grouped_P28.index
            mean_values_P28 = grouped_P28['Value', 'mean']
            std_values_P28 = grouped_P28['Value', f'{errorbar}']            
            # Plot the line plot with error bars
            line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4)

            E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
            grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_E19 = grouped_E19['Value', 'mean']
            std_values_E19 = grouped_E19['Value', f'{errorbar}']      
            line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
            
            P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
            grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P02 = grouped_P02['Value', 'mean']
            std_values_P02 = grouped_P02['Value', f'{errorbar}']      
            line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)    

            P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
            grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P10 = grouped_P10['Value', 'mean']
            std_values_P10 = grouped_P10['Value', f'{errorbar}']      
            line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)       

            P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
            grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P20 = grouped_P20['Value', 'mean']
            std_values_P20 = grouped_P20['Value', f'{errorbar}']      
            line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
            
            if y_norm == True:
                ymin = math.floor(data_melted['Value'].min())
                axs[i, j].set_ylim(ymin)
               
            # if ano2w == True:
            #     model = ols('Value ~ Time + Group + Time:Group', data=data_melted).fit()
            #     result = sm.stats.anova_lm(model, type=2)
            #     p2wano = result['PR(>F)']['Group']                

    
            # # Add a title and axis labels            
            axs[i, j].ticklabel_format(axis='y', style=style, scilimits=(0,0), useMathText=True, useOffset=True) # force scientific notation of all y labels      
            axs[i, j].tick_params(axis='y', which='major', pad=pad, width=1) # move lables closer to y ticks                        
            axs[i, j].set_xticks([0, 4, 8, 12, 16, 20, 24])
            axs[i, j].tick_params(axis='x', which='major', pad=0, width=1)
            # axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_linewidth(1)
            axs[i, j].spines['left'].set_linewidth(1)    
            if describe == True:
                des_name = des_list[counter]
                axs[i, j].set_title(f'{name} {des_name}', pad=-5, y=1.001)
            else:
                axs[i, j].set_title(f'{name}',  pad=-5, y=1.001)  # pad does not work on bottom plots, unless y=1.001 workaround
            
            # needs work
            # ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
            [[line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28']]
            
            # pvalues
            if tissue == 'SCN':
                pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))  
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))

            if tissue == 'PLS':
                pC = str(round(float(data.loc[data.Name == name, 'E19_PLS_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_PLS_emp p BH Corrected']), 4))  
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
                
            if tissue == 'LIV':
                pC = str(round(float(data.loc[data.Name == name, 'E19_LIV_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_LIV_emp p BH Corrected']), 4))  
    
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
                    
                    
            counter += 1
           
    plt.suptitle(f"{title}") 
    plt.savefig(f'Traces_{title} {tissue}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'Traces_{title} {tissue}.svg', format = 'svg', bbox_inches = 'tight')  
    plt.show()
    plt.clf()
    plt.close()   



# this version makes full lines for E19 and P28 profiles with p<0.05 as requested by a reviewer
def plot_gene_group_select_tissue_review(data, name_list, title, tissue='SCN', annotation=True, describe=False, des_list=[], y_norm=True, errorbar='sem', fnt=80, ano2w=True, style='plain', pad=-2):  # errorbar='std' style='sci'
    
    ### To save as vector svg with fonts editable in Corel ###
    # mpl.use('svg')                                                                          #import matplotlib as mpl
    # new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    # mpl.rcParams.update(new_rc_params)

    counter = 0
    Nr = len(name_list)    
    Nc = math.sqrt(Nr) #find the square root of x
    Nw = int(Nr/Nc)
    if Nc.is_integer() == False: #check y is not a whole number
        Nw = math.ceil(Nc)
        Nc = int(round(Nc, 0))        
    Nc = int(Nc)
    figsize = (Nc*3, Nw*3)
    
    fig, axs = plt.subplots(Nw, Nc, sharex=True, figsize=figsize)
    for i in range(Nw):
        for j in range(Nc):                
            
            if Nw*Nc > Nr:
                if counter == Nr:
                    break  
                            
            df=pd.DataFrame()
            # df['Name'] = data['Name']            
            # df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']  
            if tissue == 'SCN':
                df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
            if tissue == 'PLS':
                df = data.loc[:, 'log2_PLS_E19_ZT00_1':'log2_PLS_P28_ZT24_5']      
            if tissue == 'LIV':
                df = data.loc[:, 'log2_LIV_E19_ZT00_1':'log2_LIV_P28_ZT24_5']  

            df.insert(0, 'Name', data['Name'])             
            name = name_list[counter]
            # Select columns with the specified data label
            dfn = df.loc[df['Name'] == name, df.columns[1:]]               
            
            # Reshape the data into long format using melt
            data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
            # Split the Timepoint column into two separate columns for time and replicate 
            # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
            data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
            data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
            # Convert Time and Replicate columns to numeric types
            data_melted["Time"] = pd.to_numeric(data_melted["Time"])
            data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])      

            # pvalues
            if tissue == 'SCN':
                pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))  
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))

            if tissue == 'PLS':
                pC = str(round(float(data.loc[data.Name == name, 'E19_PLS_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_PLS_emp p BH Corrected']), 4))  
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
                
            if tissue == 'LIV':
                pC = str(round(float(data.loc[data.Name == name, 'E19_LIV_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_LIV_emp p BH Corrected']), 4))  
    
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))


            P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
            # Group the data by 'time' and calculate the mean and standard deviation
            grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            # Extract the values for plotting
            time_points = grouped_P28.index
            mean_values_P28 = grouped_P28['Value', 'mean']
            std_values_P28 = grouped_P28['Value', f'{errorbar}']            
            # Plot the line plot with error bars
            if float(pX) < 0.05:
                line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4) # full line
            else:
                line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, ls='--', capsize=4)

            E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
            grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_E19 = grouped_E19['Value', 'mean']
            std_values_E19 = grouped_E19['Value', f'{errorbar}']
            if float(pC) < 0.05:
                line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, capsize=4) # full line
            else:
                line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
            
            P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
            grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P02 = grouped_P02['Value', 'mean']
            std_values_P02 = grouped_P02['Value', f'{errorbar}']      
            line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)    

            P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
            grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P10 = grouped_P10['Value', 'mean']
            std_values_P10 = grouped_P10['Value', f'{errorbar}']      
            line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)       

            P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
            grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P20 = grouped_P20['Value', 'mean']
            std_values_P20 = grouped_P20['Value', f'{errorbar}']      
            line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
            
            if y_norm == True:
                ymin = math.floor(data_melted['Value'].min())
                axs[i, j].set_ylim(ymin)
               
            # if ano2w == True:
            #     model = ols('Value ~ Time + Group + Time:Group', data=data_melted).fit()
            #     result = sm.stats.anova_lm(model, type=2)
            #     p2wano = result['PR(>F)']['Group']                

    
            # # Add a title and axis labels            
            axs[i, j].ticklabel_format(axis='y', style=style, scilimits=(0,0), useMathText=True, useOffset=True) # force scientific notation of all y labels      
            axs[i, j].tick_params(axis='y', which='major', pad=pad, width=1) # move lables closer to y ticks                        
            axs[i, j].set_xticks([0, 4, 8, 12, 16, 20, 24])
            axs[i, j].tick_params(axis='x', which='major', pad=0, width=1)
            # axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_linewidth(1)
            axs[i, j].spines['left'].set_linewidth(1)    
            if describe == True:
                des_name = des_list[counter]
                axs[i, j].set_title(f'{name} {des_name}', pad=-5, y=1.001)
            else:
                axs[i, j].set_title(f'{name}',  pad=-5, y=1.001)  # pad does not work on bottom plots, unless y=1.001 workaround
            
            # needs work
            # ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
            [[line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28']]
                               
                    
            counter += 1
           
    plt.suptitle(f"{title}") 
    plt.savefig(f'Traces_{title} {tissue}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'Traces_{title} {tissue}.svg', format = 'svg', bbox_inches = 'tight')  
    plt.show()
    plt.clf()
    plt.close()  



# this version makes full lines for all SCN age profiles with p<0.05. May cause problem because LIV and PLS P02-P20 eJTK data are missing!
def plot_gene_group_select_tissue_new(data, name_list, title, tissue='SCN', annotation=True, describe=False, des_list=[], y_norm=True, errorbar='sem', fnt=80, ano2w=True, style='plain', pad=-2):  # errorbar='std' style='sci'
    
    ### To save as vector svg with fonts editable in Corel ###
    # mpl.use('svg')                                                                          #import matplotlib as mpl
    # new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    # mpl.rcParams.update(new_rc_params)

    counter = 0
    Nr = len(name_list)    
    Nc = math.sqrt(Nr) #find the square root of x
    Nw = int(Nr/Nc)
    if Nc.is_integer() == False: #check y is not a whole number
        Nw = math.ceil(Nc)
        Nc = int(round(Nc, 0))        
    Nc = int(Nc)
    figsize = (Nc*3, Nw*3)
    
    fig, axs = plt.subplots(Nw, Nc, sharex=True, figsize=figsize)
    for i in range(Nw):
        for j in range(Nc):                
            
            if Nw*Nc > Nr:
                if counter == Nr:
                    break  
                            
            df=pd.DataFrame()
            # df['Name'] = data['Name']            
            # df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']  
            if tissue == 'SCN':
                df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
            if tissue == 'PLS':
                df = data.loc[:, 'log2_PLS_E19_ZT00_1':'log2_PLS_P28_ZT24_5']      
            if tissue == 'LIV':
                df = data.loc[:, 'log2_LIV_E19_ZT00_1':'log2_LIV_P28_ZT24_5']  

            df.insert(0, 'Name', data['Name'])             
            name = name_list[counter]
            # Select columns with the specified data label
            dfn = df.loc[df['Name'] == name, df.columns[1:]]               
            
            # Reshape the data into long format using melt
            data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
            # Split the Timepoint column into two separate columns for time and replicate 
            # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
            data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
            data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
            # Convert Time and Replicate columns to numeric types
            data_melted["Time"] = pd.to_numeric(data_melted["Time"])
            data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])      

            # pvalues
            if tissue == 'SCN':
                pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))  
                p02 = str(round(float(data.loc[data.Name == name, 'P02_emp p BH Corrected']), 4))
                p10 = str(round(float(data.loc[data.Name == name, 'P10_emp p BH Corrected']), 4))
                p20 = str(round(float(data.loc[data.Name == name, 'P20_emp p BH Corrected']), 4))
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))

            if tissue == 'PLS':
                pC = str(round(float(data.loc[data.Name == name, 'E19_PLS_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_PLS_emp p BH Corrected']), 4))  
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
                
            if tissue == 'LIV':
                pC = str(round(float(data.loc[data.Name == name, 'E19_LIV_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_LIV_emp p BH Corrected']), 4))  
    
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))


            P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
            # Group the data by 'time' and calculate the mean and standard deviation
            grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            # Extract the values for plotting
            time_points = grouped_P28.index
            mean_values_P28 = grouped_P28['Value', 'mean']
            std_values_P28 = grouped_P28['Value', f'{errorbar}']            
            # Plot the line plot with error bars
            if float(pX) < 0.05:
                line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4) # full line
            else:
                line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, ls='--', capsize=4)

            E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
            grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_E19 = grouped_E19['Value', 'mean']
            std_values_E19 = grouped_E19['Value', f'{errorbar}']
            if float(pC) < 0.05:
                line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, capsize=4) # full line
            else:
                line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
            
            P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
            grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P02 = grouped_P02['Value', 'mean']
            std_values_P02 = grouped_P02['Value', f'{errorbar}']      
            if float(p02) < 0.05:
                line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, capsize=4) # full line
            else:
                line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)  

            P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
            grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P10 = grouped_P10['Value', 'mean']
            std_values_P10 = grouped_P10['Value', f'{errorbar}']      
            if float(p10) < 0.05:
                line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, capsize=4)  # full line
            else:
                line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)   

            P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
            grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P20 = grouped_P20['Value', 'mean']
            std_values_P20 = grouped_P20['Value', f'{errorbar}']      
            if float(p20) < 0.05:
                line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, capsize=4)   # full line
            else:
                line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
            
            if y_norm == True:
                ymin = math.floor(data_melted['Value'].min())
                axs[i, j].set_ylim(ymin)
               
            # if ano2w == True:
            #     model = ols('Value ~ Time + Group + Time:Group', data=data_melted).fit()
            #     result = sm.stats.anova_lm(model, type=2)
            #     p2wano = result['PR(>F)']['Group']                
    
            # # Add a title and axis labels            
            axs[i, j].ticklabel_format(axis='y', style=style, scilimits=(0,0), useMathText=True, useOffset=True) # force scientific notation of all y labels      
            axs[i, j].tick_params(axis='y', which='major', pad=pad, width=1) # move lables closer to y ticks                        
            axs[i, j].set_xticks([0, 4, 8, 12, 16, 20, 24])
            axs[i, j].tick_params(axis='x', which='major', pad=0, width=1)
            # axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_linewidth(1)
            axs[i, j].spines['left'].set_linewidth(1)    
            if describe == True:
                des_name = des_list[counter]
                axs[i, j].set_title(f'{name} {des_name}', pad=-5, y=1.001)
            else:
                axs[i, j].set_title(f'{name}',  pad=-5, y=1.001)  # pad does not work on bottom plots, unless y=1.001 workaround
            
            # needs work
            # ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
            [[line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28']]
                               
            counter += 1
           
    plt.suptitle(f"{title}") 
    plt.savefig(f'Traces_{title} {tissue}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'Traces_{title} {tissue}.svg', format = 'svg', bbox_inches = 'tight')  
    plt.show()
    plt.clf()
    plt.close()  


# annotated Volcano plot - need precalculated logFC and FDR pval values from RNAlysis deseq2 or similar and merged with main data
def volcano_annotated(data, logFC_col, pval_col, mydir, logFC_cutoff=2, pval_cutoff=2, save_list = True):
    
    plt.subplots(figsize=(4, 4))
    # Volcano plot - data has col with log Fold Changes in logFC, EDGE-R or DGE2 pVals in adj.P.Val
    plt.scatter(x=data[logFC_col],y=data[pval_col].apply(lambda x:-np.log10(x)),s=1,label="Not significant")

    # highlight down- or up- regulated genes
    up = data[(data[logFC_col]<=- logFC_cutoff)&(data[pval_col]<= pval_cutoff)]
    down = data[(data[logFC_col]>= logFC_cutoff)&(data[pval_col]<= pval_cutoff)]
    
    # list of up and downreg - change names of columns according to data
    uplist = data[(data[logFC_col]<=- logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]
    downlist = data[(data[logFC_col]>= logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]    

    plt.scatter(x=down[logFC_col],y=down[pval_col].apply(lambda x:-np.log10(x)),s=3,label="Down-regulated",color="orange")
    plt.scatter(x=up[logFC_col],y=up[pval_col].apply(lambda x:-np.log10(x)),s=3,label="Up-regulated",color="green")

    # add text index to significant values (to nicely adjust, need from adjustText import adjust_text, see www.hemtools.readthedocs.io)
    for i,r in up.iterrows():
        plt.annotate(r[0], xy=(r[logFC_col], -np.log10(r[pval_col])), fontsize=7)
    for i,r in down.iterrows():
        plt.annotate(r[0], xy=(r[logFC_col], -np.log10(r[pval_col])), fontsize=7)

    plt.xlabel(logFC_col)
    plt.ylabel("-logFDR")
    plt.axvline(-2,color="grey",linestyle="--")
    plt.axvline(2,color="grey",linestyle="--")
    plt.axhline(2,color="grey",linestyle="--")
    plt.legend()

    plt.savefig(f'Volcano {logFC_col}.svg', format = 'svg', bbox_inches = 'tight')
    plt.savefig(f'Volcano {logFC_col}.png', format = 'png', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()
    
    if save_list is True:
        with open (f'{mydir}\\Volcano {logFC_col} list.txt', 'a', encoding="utf-8") as f: f.write(f'Upregulated: \n{uplist}\n\nDownregulated: \n{downlist}') 
        uplist.to_csv(f'Volcano {logFC_col} upregulated.csv')
        downlist.to_csv(f'Volcano {logFC_col} downregulated.csv')

##################### Tkinter button for browse/set_dir ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button


root = Tk()
folder_path = StringVar()
lbl1 = Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = Button(text="Browse folder", command=browse_button)
buttonBrowse.grid()
mainloop()

mydir = os.getcwd()

data_raw = pd.read_csv('SCN_raw_data_2024.csv', delimiter = ',', encoding = "utf-8", low_memory=False)
# prepare for transposition later
data_filt = data_raw.rename(columns={'Metabolite name (full)': "sample"}).set_index('sample').iloc[:, 7:]


# check missing values
total_missing_SCN = data_filt.iloc[:851, 0:175].isna().sum().sum() + (data_filt.iloc[:, 0:175] == 0).sum().sum()
total_missing_PLS = data_filt.iloc[:851, 175:350].isna().sum().sum() + (data_filt.iloc[:, 175:350] == 0).sum().sum()

# for SCN or PLS only
# data_filt = data_filt.iloc[:851, 175:350]
# data_filt = data_filt.iloc[:851, 0:175]

total_values = data_filt.count().sum()
print(f'Total values {total_values}')

# Here I impute if 2 or less out of 5 have 0, otherwise I add NaNs and remove, chatGPT code
# Iterate over the DataFrame in chunks of 5 columns
group_size = 5
total_filled_count1 = 0  # Initialize counter

for i in range(0, data_filt.shape[1], group_size):
    cols = data_filt.columns[i:i + group_size]
    block = data_filt.loc[:, cols]

    # Identify rows where there are 2 or fewer zero values
    rows_with_few_zeros = (block == 0).sum(axis=1) <= 2
    few_zero_indices = rows_with_few_zeros[rows_with_few_zeros].index

    # Subset of DataFrame to modify
    sub_df = data_filt.loc[few_zero_indices, cols]

    # Count zeros before replacing with NaN (only in selected rows)
    zeros_to_replace = (sub_df == 0).sum().sum()

    # Replace 0 with NaN in those rows
    sub_df_replaced = sub_df.replace(0.0, np.nan)

    # Compute row medians
    row_medians = sub_df_replaced.median(axis=1)

    # Fill NaNs with row medians
    sub_df_filled = sub_df_replaced.apply(lambda row: row.fillna(row_medians[row.name]), axis=1)

    # Count how many NaNs were filled (should equal the number of zeros replaced)
    nan_filled = sub_df_replaced.isna().sum().sum() - sub_df_filled.isna().sum().sum()

    # Update counter
    total_filled_count1 += nan_filled

    # Write back the filled data
    data_filt.loc[few_zero_indices, cols] = sub_df_filled

    # Set remaining rows (with >2 zeros) to NaN
    many_zero_indices = rows_with_few_zeros[~rows_with_few_zeros].index
    data_filt.loc[many_zero_indices, cols] = np.nan

print(f"First number of 0 values filled with row medians: {total_filled_count1}")


data_filt2 = data_filt.copy()   
                                                                                       

group_size = 35
total_filled_count = 0  # Counter for filled NaNs

for i in range(0, data_filt.shape[1], group_size):
    # Select the next group of 35 columns, skipping the last 5 rows as in your code
    cols = data_filt.columns[i:i + group_size]
    block = data_filt.iloc[:-5, i:i + group_size]

    # Identify rows with 10 or fewer NaNs (i.e., 20% or less)
    rows_with_few_zeros = block.isna().sum(axis=1) <= 10
    few_zero_indices = rows_with_few_zeros[rows_with_few_zeros].index

    # Extract just the rows/columns to be filled
    sub_df = data_filt.loc[few_zero_indices, cols]

    # Count NaNs before imputation
    nan_before = sub_df.isna().sum().sum()

    # Compute row medians
    row_medians = sub_df.median(axis=1)

    # Fill NaNs with row medians
    sub_df_filled = sub_df.apply(lambda row: row.fillna(row_medians[row.name]), axis=1)

    # Count NaNs after imputation
    nan_after = sub_df_filled.isna().sum().sum()

    # Update filled count
    total_filled_count += (nan_before - nan_after)

    # Write back the filled data
    data_filt.loc[few_zero_indices, cols] = sub_df_filled

    # Set remaining rows (with >10 NaNs) to NaN
    many_zero_indices = rows_with_few_zeros[~rows_with_few_zeros].index
    data_filt.loc[many_zero_indices, cols] = np.nan

print(f"Second number of NaN values filled with row medians: {total_filled_count}")

print(f"Total filled values: {total_filled_count1 + total_filled_count}")

print(f"Total valid values: {data_filt.count().sum()}")

print(f"Percentage of imputed values: {((total_filled_count1 + total_filled_count) / data_filt.count().sum()) * 100}")

# check distribution of data
plt.hist(data_filt.values.flatten(),bins=500)


data_filt3 = data_filt.copy()
data_filt3.to_csv('python_data_filt_withNaN.csv')


after_imputation_missing_SCN = data_filt.iloc[:851, 0:175].isna().sum().sum() + (data_filt.iloc[:, 0:175] == 0).sum().sum()
after_imputation_missing_PLS = data_filt.iloc[:851, 175:350].isna().sum().sum() + (data_filt.iloc[:, 175:350] == 0).sum().sum()


# add 0 instead of NaNs
data_filt.iloc[:-5, :] = data_filt.iloc[:-5, :].fillna(0)
data_filt.to_csv('python_data_filt_with0.csv')

# add 1 to all data to remove 0 before log, but this creates problems and artificial rhythm
data_filt.iloc[:-5, :] = data_filt.iloc[:-5, :] + 1
imputed_mat = data_filt.T
# Log transform the data without qPCR
log2_mat = np.log2(imputed_mat.iloc[:, :-5])
log2_mat = pd.concat([log2_mat, imputed_mat.iloc[:, -5:]], axis=1)

# Raw data
log2_mat.T.to_csv('python_log2_metabolites.csv')


# function to calculate zscores in this specific dataframe
def z_score_groups_separate_tissues(df_std, group_size=35, tissues=3):   
    df_std = df_std.copy()
    for column in range(len(df_std.columns)): 
        df_std.iloc[0*group_size:1*group_size, column] = (df_std.iloc[0*group_size:1*group_size, column] - df_std.iloc[0*group_size:1*group_size, column].mean()) / df_std.iloc[0*group_size:1*group_size, column].std()
        df_std.iloc[1*group_size:2*group_size, column] = (df_std.iloc[1*group_size:2*group_size, column] - df_std.iloc[1*group_size:2*group_size, column].mean()) / df_std.iloc[1*group_size:2*group_size, column].std()
        df_std.iloc[2*group_size:3*group_size, column] = (df_std.iloc[2*group_size:3*group_size, column] - df_std.iloc[2*group_size:3*group_size, column].mean()) / df_std.iloc[2*group_size:3*group_size, column].std()
        df_std.iloc[3*group_size:4*group_size, column] = (df_std.iloc[3*group_size:4*group_size, column] - df_std.iloc[3*group_size:4*group_size, column].mean()) / df_std.iloc[3*group_size:4*group_size, column].std()
        df_std.iloc[4*group_size:5*group_size, column] = (df_std.iloc[4*group_size:5*group_size, column] - df_std.iloc[4*group_size:5*group_size, column].mean()) / df_std.iloc[4*group_size:5*group_size, column].std()  
        
        df_std.iloc[5*group_size:6*group_size, column] = (df_std.iloc[5*group_size:6*group_size, column] - df_std.iloc[5*group_size:6*group_size, column].mean()) / df_std.iloc[5*group_size:6*group_size, column].std()
        df_std.iloc[6*group_size:7*group_size, column] = (df_std.iloc[6*group_size:7*group_size, column] - df_std.iloc[6*group_size:7*group_size, column].mean()) / df_std.iloc[6*group_size:7*group_size, column].std()
        df_std.iloc[7*group_size:8*group_size, column] = (df_std.iloc[7*group_size:8*group_size, column] - df_std.iloc[7*group_size:8*group_size, column].mean()) / df_std.iloc[7*group_size:8*group_size, column].std()
        df_std.iloc[8*group_size:9*group_size, column] = (df_std.iloc[8*group_size:9*group_size, column] - df_std.iloc[8*group_size:9*group_size, column].mean()) / df_std.iloc[8*group_size:9*group_size, column].std()
        df_std.iloc[9*group_size:10*group_size, column] = (df_std.iloc[9*group_size:10*group_size, column] - df_std.iloc[9*group_size:10*group_size, column].mean()) / df_std.iloc[9*group_size:10*group_size, column].std() 
        
        df_std.iloc[10*group_size:11*group_size, column] = (df_std.iloc[10*group_size:11*group_size, column] - df_std.iloc[10*group_size:11*group_size, column].mean()) / df_std.iloc[10*group_size:11*group_size, column].std()
        df_std.iloc[11*group_size:12*group_size, column] = (df_std.iloc[11*group_size:12*group_size, column] - df_std.iloc[11*group_size:12*group_size, column].mean()) / df_std.iloc[11*group_size:12*group_size, column].std()
        df_std.iloc[12*group_size:13*group_size, column] = (df_std.iloc[12*group_size:13*group_size, column] - df_std.iloc[12*group_size:13*group_size, column].mean()) / df_std.iloc[12*group_size:13*group_size, column].std()
        df_std.iloc[13*group_size:14*group_size, column] = (df_std.iloc[13*group_size:14*group_size, column] - df_std.iloc[13*group_size:14*group_size, column].mean()) / df_std.iloc[13*group_size:14*group_size, column].std()
        df_std.iloc[14*group_size:15*group_size, column] = (df_std.iloc[14*group_size:15*group_size, column] - df_std.iloc[14*group_size:15*group_size, column].mean()) / df_std.iloc[14*group_size:15*group_size, column].std() 
        
    return df_std


# Z-score the data  
z_score_data = z_score_groups_separate_tissues(log2_mat).T
# check for problems
log2_mat.iloc[0:35, 0]
z_score_data.iloc[0:35, 0]
z_score_data.iloc[175:210, 0]
z_score_data.iloc[350:525, 0]
# remove NaNs
z_score_data.iloc[:-5, :] = z_score_data.iloc[:-5, :].fillna(0)
z_score_data.to_csv('python_zscore_metabolites.csv')
       
    
# Standardise the data by z-score scaler, together all ages and all tissues, used mainly for PCA
processed_data = pd.DataFrame(StandardScaler().fit_transform(log2_mat), columns=log2_mat.columns, index=log2_mat.index).T
# save final data
processed_data.to_csv('python_processed_metabolites with outliers.csv')

# analyze data externally in eJTK (Biodare2) and BIOCYCLE (Circadiomics) and merge results with data
# LOAD PROCESSED DATA with RESULTS of RHYTHM ANALYSES, only use data without outliers > 3 sigma from now on
# data = pd.read_csv('_processed data with eJTK BIO analytes in index.csv', delimiter = ',', encoding = "utf-8", low_memory=False)
data_T = pd.read_csv('_processed data with eJTK BIO analytes in columns.csv', delimiter = ',', encoding = "utf-8", low_memory=False)

data = pd.read_csv('_processed SCN PLS LIV data with SCN BIO - analytes in index.csv', delimiter = ',', encoding = "utf-8", low_memory=False)

# CORRECT annotation - Lysine is not Peptide but Amino acid
data.loc[data.Name == 'Lysine', 'Class'] = 'Amino acid'

# doubleplot missing P10 ZT24 for clock genes, use .values to avoid chained indexing/slice operations problem
data.loc[851:855, 'score_P10_ZT24_1':'score_P10_ZT24_5'] = data.loc[851:855, 'score_P10_ZT00_1':'score_P10_ZT00_5'].values

# merge with BIOlog (Biocycle 24 run on log data for AMP plots) + added LIV and PLS BIOCYCLE results from raw (not log2) values
BIOlog = pd.read_csv('BIOlog complete results.csv', delimiter = ',', encoding = "utf-8", low_memory=False)
data = data.merge(BIOlog, left_on='Original_row', right_on='Original_row', how='outer')

# merge with eJTK results one age at a time
data = data.merge(pd.read_csv('eJTK_E19.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P02.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P10.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P20.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P28.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
# merge with eJTK results of LIVER and PLASMA
data = data.merge(pd.read_csv('eJTK_E19_PLS.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P28_PLS.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_E19_LIV.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')
data = data.merge(pd.read_csv('eJTK_P28_LIV.csv', delimiter = ',', encoding = "utf-8", low_memory=False), left_on='Original_row', right_on='Original_row', how='outer')

data.to_csv('_processed SCN PLS LIV data with BIO eJTK - analytes in index.csv')

# Calculate means for z-score values
group_size = 5

list_of_timepoints = ['00','00','00','00','00', '04','04','04','04','04', '08','08','08','08','08', '12','12','12','12','12', '16','16','16','16','16', '20','20','20','20','20', '24','24','24','24','24', 
                      '00','00','00','00','00', '04','04','04','04','04', '08','08','08','08','08', '12','12','12','12','12', '16','16','16','16','16', '20','20','20','20','20', '24','24','24','24','24', 
                      '00','00','00','00','00', '04','04','04','04','04', '08','08','08','08','08', '12','12','12','12','12', '16','16','16','16','16', '20','20','20','20','20', '24','24','24','24','24', 
                      '00','00','00','00','00', '04','04','04','04','04', '08','08','08','08','08', '12','12','12','12','12', '16','16','16','16','16', '20','20','20','20','20', '24','24','24','24','24', 
                      '00','00','00','00','00', '04','04','04','04','04', '08','08','08','08','08', '12','12','12','12','12', '16','16','16','16','16', '20','20','20','20','20', '24','24','24','24','24']


# Iterate over the DataFrame in chunks of 5 columns, calculate means in each SCN group separately, named 'grpz_E19_00', 'grpz_E19_04',...
list_of_columns = data.columns[15:190]
list_of_columns = [str(x) for x in list_of_columns]
for i in range(1, len(data.columns[15:190]), group_size):
    data.loc[:, f'{list_of_columns[i][0:9]}{list_of_timepoints[i-1]}'] = data.iloc[:, (i+14):(i+14+5)].mean(axis=1, numeric_only=True)
    
# dtto for PLS >> data.columns[190:365] but need to change naming to {list_of_columns[i][0:13]} and adjust numbers
list_of_columns = data.columns[190:365]
list_of_columns = [str(x) for x in list_of_columns]
for i in range(1, len(data.columns[190:365]), group_size):
    data.loc[:, f'{list_of_columns[i][0:13]}{list_of_timepoints[i-1]}'] = data.iloc[:, (i+189):(i+189+5)].mean(axis=1, numeric_only=True)

# for LIV >>> data.columns[365:540]  but need to change naming to {list_of_columns[i][0:13]} and adjust numbers
list_of_columns = data.columns[365:540]
list_of_columns = [str(x) for x in list_of_columns]
for i in range(1, len(data.columns[365:540]), group_size):
    data.loc[:, f'{list_of_columns[i][0:13]}{list_of_timepoints[i-1]}'] = data.iloc[:, (i+364):(i+364+5)].mean(axis=1, numeric_only=True)


# Iterate over the DataFrame in chunks of 5 columns, calculate means all SCN groups together, named 'E19_00', 'E19_04',...
list_of_columns = data.columns[540:715]
list_of_columns = [str(x) for x in list_of_columns]
for i in range(1, len(list_of_columns), group_size):
    data.loc[:, f'{list_of_columns[i][6:9]}_{list_of_timepoints[i-1]}'] = data.iloc[:, (i+539):(i+539+5)].mean(axis=1, numeric_only=True)

# dtto for PLS >> data.columns[715:890] but need to change naming to {list_of_columns[i][0:13]} and adjust numbers
# for LIV >>> data.columns[890:1065]  but need to change naming to {list_of_columns[i][0:13]} and adjust numbers

# Iterate over the DataFrame in chunks of 5 columns, calculate means all groups together from log for K-means, named 'log2_E19_00', 'log2_E19_04',...
list_of_columns = data.columns[1065:1240]
list_of_columns = [str(x) for x in list_of_columns]
for i in range(1, len(list_of_columns), group_size):
    data.loc[:, f'{list_of_columns[i][0:9]}{list_of_timepoints[i-1]}'] = data.iloc[:, (i+1064):(i+1064+5)].mean(axis=1, numeric_only=True)

# dtto for PLS >> data.columns[1240:1415] but need to change naming to {list_of_columns[i][0:13]} and adjust numbers
# for LIV >>> data.columns[1415:1590]  but need to change naming to {list_of_columns[i][0:13]} and adjust numbers

data = data.copy()
# switch off warnings, to avoid anova warning 
import warnings
warnings.filterwarnings("ignore")

# add one-way ANOVA for SCN
df=pd.DataFrame()
df = data.loc[:, 'raw_E19_ZT00_1':'raw_P28_ZT24_5'] # calculate ANOVA raw with outliers
df.loc[851:855, 'raw_P10_ZT24_1':'raw_P10_ZT24_5'] = df.loc[851:855, 'raw_P10_ZT00_1':'raw_P10_ZT00_5'].values
df.insert(0, 'Name', data['Name'])    # old data      
p1wano_list_E19 = []
p1wano_list_P02 = []
p1wano_list_P10 = []
p1wano_list_P20 = []
p1wano_list_P28 = []
hue_order = [0, 4, 8, 12, 16, 20, 24]
# pKW_list_E19 = []
# pKW_list_P02 = []
# pKW_list_P10 = []
# pKW_list_P20 = []
# pKW_list_P28 = []
gene_name_list = df['Name']
for i in gene_name_list:
    dfn = df.loc[df['Name'] == i, df.columns[1:]]  
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
    # Split the Timepoint column into two separate columns for time and replicate 
    # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
    data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
    data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
    # Convert Time and Replicate columns to numeric types
    data_melted["Time"] = pd.to_numeric(data_melted["Time"])
    data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"]) 
    
    #Scipy 1way anova
    E19list = []
    for j in range(len(hue_order)):
        E19list.append(data_melted['Value'][(data_melted['Group'] == 'E19')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    UC, pC = stats.f_oneway(*E19list)
    p1wano_list_E19.append(pC)
    P02list = []
    for j in range(len(hue_order)):
        P02list.append(data_melted['Value'][(data_melted['Group'] == 'P02')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    US, pS = stats.f_oneway(*P02list)
    p1wano_list_P02.append(pS)
    P10list = []
    for j in range(len(hue_order)):
        P10list.append(data_melted['Value'][(data_melted['Group'] == 'P10')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    US, pS = stats.f_oneway(*P10list)
    p1wano_list_P10.append(pS)
    P20list = []
    for j in range(len(hue_order)):
        P20list.append(data_melted['Value'][(data_melted['Group'] == 'P20')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    US, pS = stats.f_oneway(*P20list)
    p1wano_list_P20.append(pS)
    P28list = []
    for j in range(len(hue_order)):
        P28list.append(data_melted['Value'][(data_melted['Group'] == 'P28')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    US, pS = stats.f_oneway(*P28list)
    p1wano_list_P28.append(pS) 
    
    # # Kruskal-W
    # for j in range(len(hue_order)):
    #     E19list.append(data_melted['Value'][(data_melted['Group'] == 'E19')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    # UC, pC = stats.kruskal(*E19list)
    # pKW_list_E19.append(pC)
    # for j in range(len(hue_order)):
    #     P02list.append(data_melted['Value'][(data_melted['Group'] == 'P02')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    # US, pS = stats.kruskal(*P02list)
    # pKW_list_P02.append(pS)
    # for j in range(len(hue_order)):
    #     P10list.append(data_melted['Value'][(data_melted['Group'] == 'P10')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    # US, pS = stats.kruskal(*P10list)
    # pKW_list_P10.append(pS)
    # for j in range(len(hue_order)):
    #     P20list.append(data_melted['Value'][(data_melted['Group'] == 'P20')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    # US, pS = stats.kruskal(*P20list)
    # pKW_list_P20.append(pS)
    # for j in range(len(hue_order)):
    #     P28list.append(data_melted['Value'][(data_melted['Group'] == 'P28')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    # US, pS = stats.kruskal(*P28list)
    # pKW_list_P28.append(pS)  
    
# add calculated anova pvals to dataframe
data['1w_ANOVA_E19'] = p1wano_list_E19
data['1w_ANOVA_P02'] = p1wano_list_P02
data['1w_ANOVA_P10'] = p1wano_list_P10
data['1w_ANOVA_P20'] = p1wano_list_P20
data['1w_ANOVA_P28'] = p1wano_list_P28


# add ANOVA one-way PLASMA
df=pd.DataFrame()
# df = data.loc[:, 'raw_E19_ZT00_1':'raw_P28_ZT24_5'] # calculate ANOVA raw with outliers
df = data.loc[:, 'raw_PLS_E19_ZT00_1':'raw_PLS_P28_ZT24_5'] # calculate ANOVA raw with outliers
# df.loc[851:855, 'raw_P10_ZT24_1':'raw_P10_ZT24_5'] = df.loc[851:855, 'raw_P10_ZT00_1':'raw_P10_ZT00_5'].values
df.insert(0, 'Name', data['Name'])    # old data      
p1wano_list_PLS_E19 = []
p1wano_list_PLS_P28 = []
hue_order = [0, 4, 8, 12, 16, 20, 24]
gene_name_list = df['Name']
for i in gene_name_list:
    dfn = df.loc[df['Name'] == i, df.columns[1:]]  
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
    # Split the Timepoint column into two separate columns for time and replicate 
    # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
    data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
    data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
    # Convert Time and Replicate columns to numeric types
    data_melted["Time"] = pd.to_numeric(data_melted["Time"])
    data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])     
    
    PLS_E19list = []
    for j in range(len(hue_order)):
        PLS_E19list.append(data_melted['Value'][(data_melted['Group'] == 'E19')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    UC, pC = stats.f_oneway(*PLS_E19list)
    p1wano_list_PLS_E19.append(pC)    
    PLS_P28list = []
    for j in range(len(hue_order)):
        PLS_P28list.append(data_melted['Value'][(data_melted['Group'] == 'P28')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    UC, pC = stats.f_oneway(*PLS_P28list)
    p1wano_list_PLS_P28.append(pC)
    
# add calculated anova pvals to dataframe
data['1w_ANOVA_PLS_E19'] = p1wano_list_PLS_E19
data['1w_ANOVA_PLS_P28'] = p1wano_list_PLS_P28    

# add ANOVA one-way LIVER
df=pd.DataFrame()
# df = data.loc[:, 'raw_E19_ZT00_1':'raw_P28_ZT24_5'] # calculate ANOVA raw with outliers
df = data.loc[:, 'raw_LIV_E19_ZT00_1':'raw_LIV_P28_ZT24_5'] # calculate ANOVA raw with outliers
# df.loc[851:855, 'raw_P10_ZT24_1':'raw_P10_ZT24_5'] = df.loc[851:855, 'raw_P10_ZT00_1':'raw_P10_ZT00_5'].values
df.insert(0, 'Name', data['Name'])    # old data      
p1wano_list_LIV_E19 = []
p1wano_list_LIV_P28 = []
hue_order = [0, 4, 8, 12, 16, 20, 24]
# pKW_list_E19 = []
# pKW_list_P02 = []
# pKW_list_P10 = []
# pKW_list_P20 = []
# pKW_list_P28 = []
gene_name_list = df['Name']
for i in gene_name_list:
    dfn = df.loc[df['Name'] == i, df.columns[1:]]  
    # Reshape the data into long format using melt
    data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
    # Split the Timepoint column into two separate columns for time and replicate 
    # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
    data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
    data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
    # Convert Time and Replicate columns to numeric types
    data_melted["Time"] = pd.to_numeric(data_melted["Time"])
    data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"]) 
        
    LIV_E19list = []
    for j in range(len(hue_order)):
        LIV_E19list.append(data_melted['Value'][(data_melted['Group'] == 'E19')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    UC, pC = stats.f_oneway(*LIV_E19list)
    p1wano_list_LIV_E19.append(pC)    
    LIV_P28list = []
    for j in range(len(hue_order)):
        LIV_P28list.append(data_melted['Value'][(data_melted['Group'] == 'P28')][(data_melted['Time'] == hue_order[j])].dropna(how='any'))    
    UC, pC = stats.f_oneway(*LIV_P28list)
    p1wano_list_LIV_P28.append(pC)       

# add calculated anova pvals to dataframe
data['1w_ANOVA_LIV_E19'] = p1wano_list_LIV_E19
data['1w_ANOVA_LIV_P28'] = p1wano_list_LIV_P28


# switch on warning
warnings.filterwarnings("default")

# Fig. 3A
Q_list = np.arange(0.0001, 1, 0.0001)
E19_list = []
P02_list = []
P10_list = []
P20_list = []
P28_list = []
eJTK_list = []
BIOc_list = []
for q in Q_list:
    
    E19_list.append(len(data.loc[(data['E19_PLS_emp p BH Corrected'] < q) & (data['1w_ANOVA_PLS_E19'] < q), 'Metabolite name (full)']))  
    P28_list.append(len(data.loc[(data['P28_PLS_emp p BH Corrected'] < q) & (data['1w_ANOVA_PLS_P28'] < q), 'Metabolite name (full)']))

# Create a figure and an axis
fig, ax1 = plt.subplots(figsize=(4, 4))
x = Q_list
y1 = E19_list
y2 = P02_list
y3 = P10_list
y4 = P20_list
y5 = P28_list
y_diff = [j - i for i,j in zip(BIOc_list, eJTK_list)]
# Plot data on the first y-axis
ax1.plot(x, y1, color='slateblue', label='E19')
ax1.set_ylabel('rhythmic metabolites')
ax1.set_xlabel('BH corrected empirical P value')
ax1.legend(frameon=False)
# Create a second y-axis
# ax2 = ax1.twinx()
ax2 = ax1
# ax2.plot(x, y2, color='grey', label='P02')
# ax2.legend(loc='upper left', frameon=False)
# ax3 = ax1
# ax3.plot(x, y3, color='green', label='P10')
# ax3.legend(loc='upper left', frameon=False)
# ax4 = ax1
# ax4.plot(x, y4, color='orange', label='P20')
# ax4.legend(loc='upper left', frameon=False)
ax5 = ax1
ax5.plot(x, y5, color='tomato', label='P28')
ax5.legend(loc='upper left', frameon=False)

# ax6 = plt.twinx()
# ax6.plot(x, y_diff, color='yellow', label='eJTK BIO diff')
# ax6.legend(loc='upper right', frameon=False)
plt.savefig('QvaluePlot BH and anov PLS.png', format = 'png', bbox_inches = 'tight')
plt.rcParams['svg.fonttype'] = 'none' 
plt.savefig('QvaluePlot BH and anov PLS.svg', format = 'svg', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()

bestq = 0.05

biocycle_collist = ['E19_P_VALUE', 'E19_Q_VALUE', 'E19_PERIOD', 'E19_LAG',
'E19_AMPLITUDE', 'E19_OFFSET', 'E19_MEAN_PERIODICITY','E19_SCATTER']


# export clock gene statistics
data.iloc[-5:, :].set_index('Name')[['E19_emp p BH Corrected', 'P02_emp p BH Corrected', 'P10_emp p BH Corrected', 'P20_emp p BH Corrected', 'P28_emp p BH Corrected']].to_csv('_mRNA_eJTK.csv')
data.iloc[-5:, :].set_index('Name')[['1w_ANOVA_E19', '1w_ANOVA_P02','1w_ANOVA_P10', '1w_ANOVA_P20', '1w_ANOVA_P28']].to_csv('_mRNA_Anova.csv')
data.iloc[-5:, :].set_index('Name')[['E19_P_VALUE', 'P02_P_VALUE','P10_P_VALUE', 'P20_P_VALUE', 'P28_P_VALUE']].to_csv('_mRNA_Bio.csv')


# PLOT metabolites, this give nice mRNA and sensible met rhythms, but leave out BIO, plot means from each grp separately
dataE19 = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)].sort_values(by=['E19_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_E19_00':'grpz_E19_24']
dataP02 = data.iloc[:-5, :].loc[(data['P02_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P02'] < bestq)].sort_values(by=['P02_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_P02_00':'grpz_P02_24']
dataP10 = data.iloc[:-5, :].loc[(data['P10_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P10'] < bestq)].sort_values(by=['P10_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_P10_00':'grpz_P10_24']
dataP20 = data.iloc[:-5, :].loc[(data['P20_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P20'] < bestq)].sort_values(by=['P20_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_P20_00':'grpz_P20_24']
dataP28 = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < bestq)].sort_values(by=['P28_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_P28_00':'grpz_P28_24']


# mRNA check
dataE19_genes = data.iloc[-5:, :].loc[(data['E19_emp p BH Corrected'] < 0.05) & (data['1w_ANOVA_E19'] < 0.05)].sort_values(by=['E19_LAG']).set_index('Name')[['E19_00', 'E19_04', 'E19_08', 'E19_12', 'E19_16', 'E19_20', 'E19_24']]
dataP02_genes = data.iloc[-5:, :].loc[(data['P02_emp p BH Corrected'] < 0.05) & (data['1w_ANOVA_P02'] < 0.05)].sort_values(by=['P02_LAG']).set_index('Name')[['P02_00', 'P02_04', 'P02_08', 'P02_12', 'P02_16', 'P02_20', 'P02_24']]
dataP10_genes = data.iloc[-5:, :].loc[(data['P10_emp p BH Corrected'] < 0.05) & (data['1w_ANOVA_P10'] < 0.05)].sort_values(by=['P10_LAG']).set_index('Name')[['P10_00', 'P10_04', 'P10_08', 'P10_12', 'P10_16', 'P10_20', 'P10_24']] 
dataP20_genes = data.iloc[-5:, :].loc[(data['P20_emp p BH Corrected'] < 0.05) & (data['1w_ANOVA_P20'] < 0.05)].sort_values(by=['P20_LAG']).set_index('Name')[['P20_00', 'P20_04', 'P20_08', 'P20_12', 'P20_16', 'P20_20', 'P20_24']] 
dataP28_genes = data.iloc[-5:, :].loc[(data['P28_emp p BH Corrected'] < 0.05) & (data['1w_ANOVA_P28'] < 0.05)].sort_values(by=['P28_LAG']).set_index('Name')[['P28_00', 'P28_04', 'P28_08', 'P28_12', 'P28_16', 'P28_20', 'P28_24']] 

E19_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'Name']
P28_set = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Name']
E19P28_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05) & (data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Name']
E19nP28_set = data.iloc[:-5, :].loc[((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05)) & ((data['P28_emp p BH Corrected'] >= bestq) | (data['1w_ANOVA_P28'] >= 0.05)), 'Name']
P28nE19_set = data.iloc[:-5, :].loc[((data['E19_emp p BH Corrected'] >= bestq) | (data['1w_ANOVA_E19'] >= 0.05)) & ((data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05)), 'Name']

# Fig. 3B
Venn(len(E19_set), len(P28_set), len(E19P28_set), labels=('E19 bestq', 'P28 bestq'), mydir=mydir)

# old version of Fig. 3H
value_count_plots_filter_sortcounts(data.iloc[:-5, :], 'Class', pval_f1=['E19_emp p BH Corrected', 'P28_emp p BH Corrected'], mydir=mydir, pval_f2=['1w_ANOVA_E19', '1w_ANOVA_P28'],  alpha1=bestq, alpha2=bestq)



# Fig. 3C
suptitle1 = "Metabolites rhythmic at age"
titleA = "E19"
titleB = "P02"
titleC = "P10"
titleD = "P20"
titleE = "P28"
x_lab = "time"
###### Heatmap plot NEW AI version with scaled heights below each other ############
# Prepare data for heatmaps
data_list = [dataE19, dataP10, dataP10, dataP20, dataP28]  # instead of empty P02 list, doubleplot P10
titles = [titleA, titleB, titleC, titleD, titleE]

# Determine the number of rows in each dataset
num_rows_list = [data.shape[0] for data in data_list]

# Calculate the total number of rows
total_rows = sum(num_rows_list)

# Calculate the relative heights for each subplot
relative_heights = [num_rows / total_rows for num_rows in num_rows_list]

# Add a small height ratio for the colorbar
colorbar_height_ratio = 0.1
relative_heights_with_cbar = relative_heights + [colorbar_height_ratio]

# Create a figure with a GridSpec
fig = plt.figure(figsize=(3, 18))  # Adjust the width and total height as needed
gs = fig.add_gridspec(nrows=6, ncols=6, width_ratios=[4, 4, 4, 4, 4, 1], height_ratios=relative_heights_with_cbar)

# Create axes for each heatmap and the colorbar
axs = [fig.add_subplot(gs[i, :5]) for i in range(5)]
cbar_ax = fig.add_subplot(gs[5, 5])

# Plot heatmaps
for i, dt in enumerate(data_list):
    sns.heatmap(
        dt, xticklabels=[0, 4, 8, 12, 16, 20, 24],
        yticklabels=False, annot=False, cbar=False, ax=axs[i], cmap="YlGnBu"
    )

# Add colorbar to the last subplot
sns.heatmap(
    dataP28, xticklabels=[0, 4, 8, 12, 16, 20, 24],
    yticklabels=False, annot=False, cbar=True, cbar_ax=cbar_ax, ax=axs[4], cmap="YlGnBu"
)

# Set titles and labels
fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
for i, ax in enumerate(axs):
    ax.set_title(titles[i], fontsize=10, fontweight='bold')
    ax.set(xlabel='CT (h)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to avoid clipping of suptitle

plt.rcParams['svg.fonttype'] = 'none' 
plt.savefig('Heatmap by circ peak BH05 AI.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig('Heatmap by circ peak BH05 AI.png', format = 'png', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()




# PLS and LIV heatmaps
data_E19_PLS = data.iloc[:-5, :].loc[(data['E19_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_E19'] < bestq)].sort_values(by=['E19_PLS_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_PLS_E19_00':'grpz_PLS_E19_24']
data_P28_PLS = data.iloc[:-5, :].loc[(data['P28_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_P28'] < bestq)].sort_values(by=['P28_PLS_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_PLS_P28_00':'grpz_PLS_P28_24']

data_E19_LIV = data.iloc[:-5, :].loc[(data['E19_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_E19'] < bestq)].sort_values(by=['E19_LIV_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_LIV_E19_00':'grpz_LIV_E19_24']
data_P28_LIV = data.iloc[:-5, :].loc[(data['P28_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_P28'] < bestq)].sort_values(by=['P28_LIV_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_LIV_P28_00':'grpz_LIV_P28_24']

###### Multi Heatmaps ############
suptitle1 = "Metabolites rhythmic at age"
titleA = "PLASMA E19"
titleB = "PLASMA P28"
titleC = "LIVER E19"
titleD = "LIVER P28"
x_lab = "time"
###### Heatmap plot NEW AI version with scaled heights below each other ############
# Prepare data for heatmaps
data_list = [data_E19_PLS, data_P28_PLS, data_E19_LIV, data_P28_LIV]
titles = [titleA, titleB, titleC, titleD]

# Determine the number of rows in each dataset
num_rows_list = [data.shape[0] for data in data_list]

# Calculate the total number of rows
total_rows = sum(num_rows_list)

# Calculate the relative heights for each subplot
relative_heights = [num_rows / total_rows for num_rows in num_rows_list]

# Add a small height ratio for the colorbar
colorbar_height_ratio = 0.1
relative_heights_with_cbar = relative_heights + [colorbar_height_ratio]

# Create a figure with a GridSpec
fig = plt.figure(figsize=(3, 18))  # Adjust the width and total height as needed
gs = fig.add_gridspec(nrows=5, ncols=5, width_ratios=[4, 4, 4, 4, 1], height_ratios=relative_heights_with_cbar)

# Create axes for each heatmap and the colorbar
axs = [fig.add_subplot(gs[i, :4]) for i in range(4)]
cbar_ax = fig.add_subplot(gs[4, 4])

# Plot heatmaps
for i, dt in enumerate(data_list):
    sns.heatmap(
        dt, xticklabels=[0, 4, 8, 12, 16, 20, 24],
        yticklabels=False, annot=False, cbar=False, ax=axs[i], cmap="YlGnBu"
    )

# Add colorbar to the last subplot
sns.heatmap(
    dataP28, xticklabels=[0, 4, 8, 12, 16, 20, 24],
    yticklabels=False, annot=False, cbar=True, cbar_ax=cbar_ax, ax=axs[3], cmap="YlGnBu"
)

# Set titles and labels
fig.suptitle(suptitle1, fontsize=14, fontweight='bold')
for i, ax in enumerate(axs):
    ax.set_title(titles[i], fontsize=10, fontweight='bold')
    ax.set(xlabel='CT (h)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to avoid clipping of suptitle

plt.rcParams['svg.fonttype'] = 'none' 
plt.savefig('Heatmap by circ peak BH05 AI PLS LIV.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig('Heatmap by circ peak BH05 AI PLS LIV.png', format = 'png', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()



# plot metabolites like this
plot_name_ages(data, "Serotonin", log=False)
plot_name_ages_select_tissue(data, "TG 54:6 (1)", tissue='SCN', y_norm=False, errorbar='sem', log=True)

# Where are metabolites with 0 level in E19 and rhythm in P28?
data.loc[(data['raw_E19_ZT00_1'] != data['raw_E19_ZT00_1']) & (data['P28_emp p BH Corrected'] < 0.05)]

# sort by amp
data_srtC = data.sort_values('P28_emp p BH Corrected').sort_values('P28_AMPLITUDE_BIOlog', ascending=False)
data_srtC.loc[(data_srtC['P28_emp p BH Corrected'] < 0.05) & (data_srtC['1w_ANOVA_P28'] < 0.05), 'Name']


###############################################################################################
###### Compare C and X group parameters with histograms ######
###############################################################################################
####### Select data to plot ########
# No filter, amps from all, even non-rhythmic genes
# Need log AMPLITUDEs, raw has too large dynamic range
# either calculate BIO on log values, or just make log of raw amplitudes
# data.loc[:, 'E19_AMPLITUDE_BIOlog2'] = np.log2(data.loc[:, 'E19_AMPLITUDE'])

# rhythmic in E19
value_name = 'E19_AMPLITUDE'
df = data.loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05)][['E19_AMPLITUDE_BIOlog', 'P28_AMPLITUDE_BIOlog']]  
suptitle_all = f'{value_name} rhythmic in E19, FDR'

# # rhythmic in P28 + anova
# value_name = 'P28_AMPLITUDE'
# df = data.loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05)][['E19_AMPLITUDE_BIOlog', 'P28_AMPLITUDE_BIOlog']]  
# suptitle_all = f'{value_name} rhythmic in P28, FDR'

# use melt to reshape the dataframe
df_melted = df.melt(var_name='GROUP', value_name=value_name)
# extract the first character of the 'Group' column to get either 'E' or 'P'
df_melted['GROUP'] = df_melted['GROUP'].str[0]

hue = "GROUP"
hue_dat = df_melted.GROUP
hue_order = ['E', 'P']
y = value_name
x = y
x_lab = y
y_lab = "Frequency"
ylim = (0, 0.15)
xlim = (0, 2)
#suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]/8)
y_coord = ylim[1] - (ylim[1]/8)

my_pal = {hue_order[0]: "slateblue", hue_order[1]: "tomato"}
sns.set_context("paper", font_scale=0.9)
g = sns.FacetGrid(df_melted, hue=hue, hue_order=hue_order, sharex=True, palette=my_pal)
g = (g.map(sns.histplot, y, kde=True, stat='density')).set(xlim=xlim).set_axis_labels(x_lab, y_lab)

###### Calculate t test p values between hue_dat for separate categories in col_dat ######
pvalues = []
datax1 = df_melted[y][hue_dat == hue_order[0]].dropna(how='any')
datax2 = df_melted[y][hue_dat == hue_order[1]].dropna(how='any')
t, p = stats.wilcoxon(datax1.values, datax2.values)  # stats.ttest_ind
pvalues = pvalues + [p]  

######## Add calculated  p values to each subplot ##########
#for ax, title, p in zip(g.axes.flat, col_order, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
for ax, p in zip(g.axes.flat, pvalues):   #zip object iteration, titles in col_order, corr values in pearson lists
    #ax.set_title(title, pad=-8)
    ax.text(x_coord, y_coord, 'Wilcoxon \nP = ' + str(round(p, 10)), fontsize=10)

####### Labels, titles, axis limits, legend ################
ax.legend(title=None, loc='center right', fontsize='x-small')
plt.suptitle(suptitle_all)
plt.savefig(f'{suptitle_all}.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{suptitle_all}.png', format = 'png', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()


# Histograms of rhythmic metabolites
dataC = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq), 'E19_AMPLITUDE_BIOlog']
dataX = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < bestq), 'P28_AMPLITUDE_BIOlog']
TT_Histograms1(dataC, dataX, mydir, name='BH and ANOVA')


###############################################################################################
####### Polar Histogram of frequency of phases (LAG from Biocycle) ########
###############################################################################################

# custom colors for nice circular hues
circular_colors = np.array([[0.91510904, 0.55114749, 0.67037311],
   [0.91696411, 0.55081563, 0.66264366],
   [0.91870995, 0.55055664, 0.65485881],
   [0.92034498, 0.55037149, 0.64702356],
   [0.92186763, 0.55026107, 0.63914306],
   [0.92327636, 0.55022625, 0.63122259],
   [0.9245696 , 0.55026781, 0.62326754],
   [0.92574582, 0.5503865 , 0.6152834 ],
   [0.92680349, 0.55058299, 0.6072758 ],
   [0.92774112, 0.55085789, 0.59925045],
   [0.9285572 , 0.55121174, 0.59121319],
   [0.92925027, 0.551645  , 0.58316992],
   [0.92981889, 0.55215808, 0.57512667],
   [0.93026165, 0.55275127, 0.56708953],
   [0.93057716, 0.5534248 , 0.55906469],
   [0.93076407, 0.55417883, 0.55105838],
   [0.93082107, 0.55501339, 0.54307696],
   [0.93074689, 0.55592845, 0.53512681],
   [0.9305403 , 0.55692387, 0.52721438],
   [0.93020012, 0.55799943, 0.51934621],
   [0.92972523, 0.55915477, 0.51152885],
   [0.92911454, 0.56038948, 0.50376893],
   [0.92836703, 0.56170301, 0.49607312],
   [0.92748175, 0.56309471, 0.48844813],
   [0.9264578 , 0.56456383, 0.48090073],
   [0.92529434, 0.56610951, 0.47343769],
   [0.92399062, 0.56773078, 0.46606586],
   [0.92254595, 0.56942656, 0.45879209],
   [0.92095971, 0.57119566, 0.4516233 ],
   [0.91923137, 0.5730368 , 0.44456642],
   [0.91736048, 0.57494856, 0.4376284 ],
   [0.91534665, 0.57692945, 0.43081625],
   [0.91318962, 0.57897785, 0.42413698],
   [0.91088917, 0.58109205, 0.41759765],
   [0.90844521, 0.58327024, 0.41120533],
   [0.90585771, 0.58551053, 0.40496711],
   [0.90312676, 0.5878109 , 0.3988901 ],
   [0.90025252, 0.59016928, 0.39298143],
   [0.89723527, 0.5925835 , 0.38724821],
   [0.89407538, 0.59505131, 0.38169756],
   [0.89077331, 0.59757038, 0.37633658],
   [0.88732963, 0.60013832, 0.37117234],
   [0.88374501, 0.60275266, 0.36621186],
   [0.88002022, 0.6054109 , 0.36146209],
   [0.87615612, 0.60811044, 0.35692989],
   [0.87215369, 0.61084868, 0.352622  ],
   [0.86801401, 0.61362295, 0.34854502],
   [0.86373824, 0.61643054, 0.34470535],
   [0.85932766, 0.61926872, 0.3411092 ],
   [0.85478365, 0.62213474, 0.3377625 ],
   [0.85010767, 0.6250258 , 0.33467091],
   [0.84530131, 0.62793914, 0.3318397 ],
   [0.84036623, 0.63087193, 0.32927381],
   [0.8353042 , 0.63382139, 0.32697771],
   [0.83011708, 0.63678472, 0.32495541],
   [0.82480682, 0.63975913, 0.32321038],
   [0.81937548, 0.64274185, 0.32174556],
   [0.81382519, 0.64573011, 0.32056327],
   [0.80815818, 0.6487212 , 0.31966522],
   [0.80237677, 0.65171241, 0.31905244],
   [0.79648336, 0.65470106, 0.31872531],
   [0.79048044, 0.65768455, 0.31868352],
   [0.78437059, 0.66066026, 0.31892606],
   [0.77815645, 0.66362567, 0.31945124],
   [0.77184076, 0.66657827, 0.32025669],
   [0.76542634, 0.66951562, 0.3213394 ],
   [0.75891609, 0.67243534, 0.32269572],
   [0.75231298, 0.67533509, 0.32432138],
   [0.74562004, 0.6782126 , 0.32621159],
   [0.73884042, 0.68106567, 0.32836102],
   [0.73197731, 0.68389214, 0.33076388],
   [0.72503398, 0.68668995, 0.33341395],
   [0.7180138 , 0.68945708, 0.33630465],
   [0.71092018, 0.69219158, 0.33942908],
   [0.70375663, 0.69489159, 0.34278007],
   [0.69652673, 0.69755529, 0.34635023],
   [0.68923414, 0.70018097, 0.35013201],
   [0.6818826 , 0.70276695, 0.35411772],
   [0.67447591, 0.70531165, 0.3582996 ],
   [0.667018  , 0.70781354, 0.36266984],
   [0.65951284, 0.71027119, 0.36722061],
   [0.65196451, 0.71268322, 0.37194411],
   [0.64437719, 0.71504832, 0.37683259],
   [0.63675512, 0.71736525, 0.38187838],
   [0.62910269, 0.71963286, 0.38707389],
   [0.62142435, 0.72185004, 0.39241165],
   [0.61372469, 0.72401576, 0.39788432],
   [0.60600841, 0.72612907, 0.40348469],
   [0.59828032, 0.72818906, 0.40920573],
   [0.59054536, 0.73019489, 0.41504052],
   [0.58280863, 0.73214581, 0.42098233],
   [0.57507535, 0.7340411 , 0.42702461],
   [0.5673509 , 0.7358801 , 0.43316094],
   [0.55964082, 0.73766224, 0.43938511],
   [0.55195081, 0.73938697, 0.44569104],
   [0.54428677, 0.74105381, 0.45207286],
   [0.53665478, 0.74266235, 0.45852483],
   [0.52906111, 0.74421221, 0.4650414 ],
   [0.52151225, 0.74570306, 0.47161718],
   [0.5140149 , 0.74713464, 0.47824691],
   [0.506576  , 0.74850672, 0.48492552],
   [0.49920271, 0.74981912, 0.49164808],
   [0.49190247, 0.75107171, 0.4984098 ],
   [0.48468293, 0.75226438, 0.50520604],
   [0.47755205, 0.7533971 , 0.51203229],
   [0.47051802, 0.75446984, 0.5188842 ],
   [0.46358932, 0.75548263, 0.52575752],
   [0.45677469, 0.75643553, 0.53264815],
   [0.45008317, 0.75732863, 0.5395521 ],
   [0.44352403, 0.75816207, 0.54646551],
   [0.43710682, 0.758936  , 0.55338462],
   [0.43084133, 0.7596506 , 0.56030581],
   [0.42473758, 0.76030611, 0.56722555],
   [0.41880579, 0.76090275, 0.5741404 ],
   [0.41305637, 0.76144081, 0.58104704],
   [0.40749984, 0.76192057, 0.58794226],
   [0.40214685, 0.76234235, 0.59482292],
   [0.39700806, 0.7627065 , 0.60168598],
   [0.39209414, 0.76301337, 0.6085285 ],
   [0.38741566, 0.76326334, 0.6153476 ],
   [0.38298304, 0.76345681, 0.62214052],
   [0.37880647, 0.7635942 , 0.62890454],
   [0.37489579, 0.76367593, 0.63563704],
   [0.37126045, 0.76370246, 0.64233547],
   [0.36790936, 0.76367425, 0.64899736],
   [0.36485083, 0.76359176, 0.6556203 ],
   [0.36209245, 0.76345549, 0.66220193],
   [0.359641  , 0.76326594, 0.66873999],
   [0.35750235, 0.76302361, 0.67523226],
   [0.35568141, 0.76272903, 0.68167659],
   [0.35418202, 0.76238272, 0.68807086],
   [0.3530069 , 0.76198523, 0.69441305],
   [0.35215761, 0.7615371 , 0.70070115],
   [0.35163454, 0.76103888, 0.70693324],
   [0.35143685, 0.76049114, 0.71310742],
   [0.35156253, 0.75989444, 0.71922184],
   [0.35200839, 0.75924936, 0.72527472],
   [0.3527701 , 0.75855647, 0.73126429],
   [0.3538423 , 0.75781637, 0.73718884],
   [0.3552186 , 0.75702964, 0.7430467 ],
   [0.35689171, 0.75619688, 0.74883624],
   [0.35885353, 0.75531868, 0.75455584],
   [0.36109522, 0.75439565, 0.76020396],
   [0.36360734, 0.75342839, 0.76577905],
   [0.36637995, 0.75241752, 0.77127961],
   [0.3694027 , 0.75136364, 0.77670417],
   [0.37266493, 0.75026738, 0.7820513 ],
   [0.37615579, 0.74912934, 0.78731957],
   [0.37986429, 0.74795017, 0.79250759],
   [0.38377944, 0.74673047, 0.797614  ],
   [0.38789026, 0.74547088, 0.80263746],
   [0.3921859 , 0.74417203, 0.80757663],
   [0.39665568, 0.74283455, 0.81243022],
   [0.40128912, 0.74145908, 0.81719695],
   [0.406076  , 0.74004626, 0.82187554],
   [0.41100641, 0.73859673, 0.82646476],
   [0.41607073, 0.73711114, 0.83096336],
   [0.4212597 , 0.73559013, 0.83537014],
   [0.42656439, 0.73403435, 0.83968388],
   [0.43197625, 0.73244447, 0.8439034 ],
   [0.43748708, 0.73082114, 0.84802751],
   [0.44308905, 0.72916502, 0.85205505],
   [0.44877471, 0.72747678, 0.85598486],
   [0.45453694, 0.72575709, 0.85981579],
   [0.46036897, 0.72400662, 0.8635467 ],
   [0.4662644 , 0.72222606, 0.86717646],
   [0.47221713, 0.72041608, 0.87070395],
   [0.47822138, 0.71857738, 0.87412804],
   [0.4842717 , 0.71671065, 0.87744763],
   [0.4903629 , 0.71481659, 0.88066162],
   [0.49649009, 0.71289591, 0.8837689 ],
   [0.50264864, 0.71094931, 0.88676838],
   [0.50883417, 0.70897752, 0.88965898],
   [0.51504253, 0.70698127, 0.89243961],
   [0.52126981, 0.70496128, 0.8951092 ],
   [0.52751231, 0.70291829, 0.89766666],
   [0.53376652, 0.70085306, 0.90011093],
   [0.54002912, 0.69876633, 0.90244095],
   [0.54629699, 0.69665888, 0.90465565],
   [0.55256715, 0.69453147, 0.90675397],
   [0.55883679, 0.69238489, 0.90873487],
   [0.56510323, 0.69021993, 0.9105973 ],
   [0.57136396, 0.68803739, 0.91234022],
   [0.57761655, 0.68583808, 0.91396258],
   [0.58385872, 0.68362282, 0.91546336],
   [0.59008831, 0.68139246, 0.91684154],
   [0.59630323, 0.67914782, 0.9180961 ],
   [0.60250152, 0.67688977, 0.91922603],
   [0.60868128, 0.67461918, 0.92023033],
   [0.61484071, 0.67233692, 0.921108  ],
   [0.62097809, 0.67004388, 0.92185807],
   [0.62709176, 0.66774097, 0.92247957],
   [0.63318012, 0.66542911, 0.92297153],
   [0.63924166, 0.66310923, 0.92333301],
   [0.64527488, 0.66078227, 0.92356308],
   [0.65127837, 0.65844919, 0.92366082],
   [0.65725076, 0.65611096, 0.92362532],
   [0.66319071, 0.65376857, 0.92345572],
   [0.66909691, 0.65142302, 0.92315115],
   [0.67496813, 0.64907533, 0.92271076],
   [0.68080311, 0.64672651, 0.92213374],
   [0.68660068, 0.64437763, 0.92141929],
   [0.69235965, 0.64202973, 0.92056665],
   [0.69807888, 0.6396839 , 0.91957507],
   [0.70375724, 0.63734122, 0.91844386],
   [0.70939361, 0.63500279, 0.91717232],
   [0.7149869 , 0.63266974, 0.91575983],
   [0.72053602, 0.63034321, 0.91420578],
   [0.72603991, 0.62802433, 0.9125096 ],
   [0.7314975 , 0.62571429, 0.91067077],
   [0.73690773, 0.62341425, 0.9086888 ],
   [0.74226956, 0.62112542, 0.90656328],
   [0.74758193, 0.61884899, 0.90429382],
   [0.75284381, 0.6165862 , 0.90188009],
   [0.75805413, 0.61433829, 0.89932181],
   [0.76321187, 0.6121065 , 0.89661877],
   [0.76831596, 0.6098921 , 0.89377082],
   [0.77336536, 0.60769637, 0.89077786],
   [0.77835901, 0.6055206 , 0.88763988],
   [0.78329583, 0.6033661 , 0.88435693],
   [0.78817477, 0.60123418, 0.88092913],
   [0.79299473, 0.59912616, 0.87735668],
   [0.79775462, 0.59704339, 0.87363986],
   [0.80245335, 0.59498722, 0.86977904],
   [0.8070898 , 0.592959  , 0.86577468],
   [0.81166284, 0.5909601 , 0.86162732],
   [0.81617134, 0.5889919 , 0.8573376 ],
   [0.82061414, 0.58705579, 0.85290625],
   [0.82499007, 0.58515315, 0.84833413],
   [0.82929796, 0.58328538, 0.84362217],
   [0.83353661, 0.58145389, 0.83877142],
   [0.8377048 , 0.57966009, 0.83378306],
   [0.8418013 , 0.57790538, 0.82865836],
   [0.84582486, 0.57619119, 0.82339871],
   [0.84977422, 0.57451892, 0.81800565],
   [0.85364809, 0.57289   , 0.8124808 ],
   [0.85744519, 0.57130585, 0.80682595],
   [0.86116418, 0.56976788, 0.80104298],
   [0.86480373, 0.56827749, 0.79513394],
   [0.86836249, 0.56683612, 0.789101  ],
   [0.87183909, 0.56544515, 0.78294645],
   [0.87523214, 0.56410599, 0.77667274],
   [0.87854024, 0.56282002, 0.77028247],
   [0.88176195, 0.56158863, 0.76377835],
   [0.88489584, 0.56041319, 0.75716326],
   [0.88794045, 0.55929505, 0.75044023],
   [0.89089432, 0.55823556, 0.74361241],
   [0.89375596, 0.55723605, 0.73668312],
   [0.89652387, 0.55629781, 0.72965583],
   [0.89919653, 0.55542215, 0.72253414],
   [0.90177242, 0.55461033, 0.71532181],
   [0.90425   , 0.55386358, 0.70802274],
   [0.90662774, 0.55318313, 0.70064098],
   [0.90890408, 0.55257016, 0.69318073],
   [0.91107745, 0.55202582, 0.68564633],
   [0.91314629, 0.55155124, 0.67804225]])

#continous color maps
#cmap="viridis"
#cmap="YlGnBu"
#cmap= grayscale_cmap(cmap)
#other circular color maps
#cmap = mpl.colors.ListedColormap(sns.hls_palette(256))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256, .33, .85, .6))
#cmap = mpl.colors.ListedColormap(sns.husl_palette(256))
#cmap = mpl.colors.ListedColormap(circular_colors)

def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r

# stackoverflow filter outliers - change m as needed (2 is default, 10 filters only most extreme)
def reject_outliers(data, m=10.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

# polar histogram function for Fig. 3-6
def polar_histogram_dual(data_Ar, data_Br, titlelist, pval_col, phase_col, filename, mydir):
    # TWO POLAR HISTO NEXT to each other 
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), sharey=True)  # figsize=(8, 8)
    data = [data_Ar, data_Br]
    # for t in range(len(titlelist)):  
    for t, d in zip(range(len(titlelist)), data): 

        outlier_reindex = ~(np.isnan(d[f'{titlelist[t]}{pval_col}']))
        data_filt = d[d.columns[:].tolist()][outlier_reindex]
        phaseseries = data_filt[f'{titlelist[t]}{phase_col}'].values.flatten()
        
        # POSITION (PHASE)
        phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
        N_bins = 23                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
        colorcode = circular_colors[0::int(round(len(circular_colors) / N_bins, 0))]   # return every 5th item from circular_colors array to get cca. 47 distinct color similar to cmap

        phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
        theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
        width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

        ax[t].bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre
        ax[t].set_yticklabels([])          # this deletes radial ticks
        ax[t].set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
        ax[t].set_theta_direction(-1)      #reverse direction of theta increases
        ax[t].set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontsize=12)  #set theta grids and labels, **kwargs for text properties
        ax[t].set_xlabel("Circadian phase (h)", fontsize=12)
        ax[t].yaxis.grid(False)   # turns off circles
        ax[t].xaxis.grid(False)  # turns off radial grids
        ax[t].tick_params(pad=-20)   # moves labels closer or further away from subplots
        ax[t].set_title(titlelist[t][:3], fontsize=16, fontweight='bold') # adjust title according to column name

        # calculate vector sum of angles and plot "Rayleigh" vector
        a_cos = map(lambda x: math.cos(x), phase)
        a_sin = map(lambda x: math.sin(x), phase)
        uv_x = sum(a_cos)/len(phase)
        uv_y = sum(a_sin)/len(phase)
        uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
        uv_phase = np.angle(complex(uv_x, uv_y))
        v_angle = uv_phase           
        v_length = uv_radius*max(phase_hist)
        #add arrow
        ax[t].annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black'))
        
    ### To save as vector svg with fonts editable in Corel ###
    plt.savefig('Polar_Histogram_Phase_both {filename}.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
    ### To save as bitmap png for easy viewing ###
    plt.savefig('Polar_Histogram_Phase_both {filename}.png', bbox_inches = 'tight')
    plt.show()
    plt.clf()
    plt.close()
        
# PLS       
data_Ar = data.iloc[:-5, :].loc[(data['E19_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_E19'] < bestq)]
data_Br = data.iloc[:-5, :].loc[(data['P28_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_P28'] < bestq)]
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)

# LIV
data_Ar = data.iloc[:-5, :].loc[(data['E19_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_E19'] < bestq)]
data_Br = data.iloc[:-5, :].loc[(data['P28_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_P28'] < bestq)]
polar_histogram_dual(data_Ar, data_Br, ['E19_LIV_', 'P28_LIV_'], 'Q_VALUE', 'LAG', 'LIV', mydir)

# SCN
data_Ar = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)]
data_Br = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < bestq)]
polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)



#######################################################################################################################################
######### Processed Data Loading - Metabolites rhythmic in C or S groups, sorted by phase in C or S group #############
#######################################################################################################################################

# use means in individual grps
dataE19 = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)].sort_values(by=['E19_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_E19_00':'grpz_E19_24']
dataP28 = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < bestq)].sort_values(by=['P28_P. Circ. Peak']).set_index('Name').loc[:, 'grpz_P28_00':'grpz_P28_24']
# Export for hierarchical clustering in RNAlysis, already normalized
dataE19.columns = [0, 4, 8, 12, 16, 20, 24]
dataP28.columns = [0, 4, 8, 12, 16, 20, 24]
dataE19.to_csv('dataE19.csv')
dataP28.to_csv('dataP28.csv')

# use means in from all ages together
dataE19_all = data.iloc[:-5, :].loc[:, 'log2_E19_00':'log2_E19_24']
dataP02_all = data.iloc[:-5, :].loc[:, 'log2_P02_00':'log2_P02_24']
dataP10_all = data.iloc[:-5, :].loc[:, 'log2_P10_00':'log2_P10_24']
dataP20_all = data.iloc[:-5, :].loc[:, 'log2_P20_00':'log2_P20_24']
dataP28_all = data.iloc[:-5, :].loc[:, 'log2_P28_00':'log2_P28_24']
# Export for hierarchical clustering in RNAlysis, already normalized
dataE19_all.columns = [0, 4, 8, 12, 16, 20, 24]
dataP02_all.columns = [i + 96 for i in dataE19_all.columns] # E21 is P0
dataP10_all.columns = [i + 192 for i in dataP02_all.columns]
dataP20_all.columns = [i + 240 for i in dataP10_all.columns]
dataP28_all.columns = [i + 192 for i in dataP20_all.columns]
data_ages_all = pd.concat([dataE19_all, dataP02_all, dataP10_all, dataP20_all, dataP28_all], axis = 1)
data_ages_all.insert(0, 'Name', data['Name'])
data_ages_all.to_csv('data_ages_all.csv')

# CLUSTERING done in RNAlysis, merge to data
clusters = pd.read_csv('clusters.csv', delimiter = ',', encoding = "utf-8", low_memory=False) 
data = data.merge(clusters, left_on = 'Name', right_on='Name', how='outer')

    
# Fig. 4 - Traces from clusters in RNAlysis
name_list = list(data.loc[data['Cluster'] == 'E19_cluster1', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'E19_cluster1', 'Class'])
plot_gene_group(data, name_list, 'E19 cluster 1', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'E19_cluster2', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'E19_cluster2', 'Class'])
plot_gene_group(data, name_list, 'E19 cluster 2', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'P28_cluster1', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'P28_cluster1', 'Class'])
plot_gene_group(data, name_list, 'P28 cluster 1', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'P28_cluster2', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'P28_cluster2', 'Class'])
plot_gene_group(data, name_list, 'P28 cluster 2', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'P28_cluster3', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'P28_cluster3', 'Class'])
plot_gene_group(data, name_list, 'P28 cluster 3', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'P28_cluster4', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'P28_cluster4', 'Class'])
plot_gene_group(data, name_list, 'P28 cluster 4', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['Cluster'] == 'P28_cluster5', 'Name'])
des_list = list(data.loc[data['Cluster'] == 'P28_cluster5', 'Class'])
plot_gene_group(data, name_list, 'P28 cluster 5', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)


# Fig. 4
value_count_plots_cluster(data, 'E19_cluster1', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'E19_cluster2', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'P28_cluster1', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'P28_cluster2', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'P28_cluster3', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'P28_cluster4', 'Class', mydir, combined=False)
value_count_plots_cluster(data, 'P28_cluster5', 'Class', mydir, combined=False)

# plot specific metabolites and genes like this

# Traces mRNA
name_list = list(data.loc[data['Class'] == 'mRNA', 'Name'])
des_list = list(data.loc[data['Class'] == 'mRNA', 'Class'])
plot_gene_group(data, name_list, 'mRNA', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

# Traces rhythmic AAs in LIV E19
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_E19'] < bestq)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_E19'] < bestq)), 'Class'])
plot_gene_group_select_tissue(data, name_list, 'Rhythmic Amino acid in E19 LIV', tissue='LIV', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# Traces rhythmic AAs in PLS E19
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_E19'] < bestq)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_E19'] < bestq)), 'Class'])
plot_gene_group_select_tissue(data, name_list, 'Rhythmic Amino acid in E19 PLS', tissue='PLS', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

# Traces rhythmic AAs in LIV P28 - None unless ANOVA adj
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_P28'] < 0.1)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_LIV_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_LIV_P28'] < 0.1)), 'Class'])
plot_gene_group_select_tissue(data, name_list, 'Rhythmic Amino acid in P28 LIV anova 0.1', tissue='LIV', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# Traces rhythmic AAs in PLS P28
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_P28'] < bestq)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_PLS_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_PLS_P28'] < bestq)), 'Class'])
plot_gene_group_select_tissue(data, name_list, 'Rhythmic Amino acid in P28 PLS', tissue='PLS', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=30, ano2w=False)

# Traces rhythmic AAs in SCN E19
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Class'])
plot_gene_group_select_tissue_new(data, name_list, 'Rhythmic Amino acid in E19 SCN', tissue='SCN', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# Traces rhythmic AAs in SCN P28 - None unless ANOVA adj
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.1)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.1)), 'Class'])
plot_gene_group_select_tissue_new(data, name_list, 'Rhythmic Amino acid in P28 SCN anova 0.1', tissue='SCN', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=15, ano2w=False)


# selected traces of PLS and SCN compounds for Fig. 5
# name_list = list(data.loc[data['Class'] == 'Amino acid', 'Name'])
name_list = ['Arginine', 'Asparagine',  'Leucine', 'Methionine', 'Phenylalanine', 'Proline', 'Threonine', 'Lysine']
des_list = name_list
plot_gene_group_select_tissue_review(data, name_list, 'Amino acid in E19 PLS corresponding to E19 SCN', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)
plot_gene_group_select_tissue_new(data, name_list, 'Amino acid in E19 SCN rhythmic almost', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)


# SCN and PLS amino acids polar plot for Fig. 5
name_list = ['Arginine', 'Asparagine',  'Leucine', 'Methionine', 'Phenylalanine', 'Proline', 'Threonine', 'Lysine']
data_Ar = data.loc[(data['Name'] == name_list[0]) | (data['Name'] == name_list[1]) | (data['Name'] == name_list[2]) | (data['Name'] == name_list[3]) | (data['Name'] == name_list[4]) | 
                   (data['Name'] == name_list[5]) | (data['Name'] == name_list[6]) | (data['Name'] == name_list[7])].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)]
data_Br = data_Ar
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)
polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)

# SCN and PLS lipids for Fig. 6
name_list = ['NAOrn 27:1;O', 'NAOrn 19:0;O', 'NAOrn 19:0;O']
des_list = name_list
plot_gene_group_select_tissue_review(data, name_list, 'Naorn in E19 PLS corresponding to E19 SCN', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)
plot_gene_group_select_tissue_new(data, name_list, 'Naorn in E19 SCN rhythmic almost', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)

name_list = ['TG 54:3', 'TG 54:6 (2)', 'TG 54:7', 'TG 56:2', 'TG 56:3', 'TG 56:8 (2)', 'TG 38:2']
des_list = name_list
plot_gene_group_select_tissue_review(data, name_list, 'diet TAG in E19 PLS corresponding to E19 SCN', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)
plot_gene_group_select_tissue_new(data, name_list, 'diet TAG E19 SCN rhythmic almost', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)

name_list = list(data.loc[(data['Class'] == 'CAR') & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Name'])
des_list = list(data.loc[(data['Class'] == 'CAR') & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Class'])
plot_gene_group_select_tissue_new(data, name_list, 'Rhythmic CAR in E19 SCN', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)
plot_gene_group_select_tissue_review(data, name_list, 'Rhythmic CAR in E19 PLS', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)

# CAR NAOrn only
name_list = ['CAR 2:0', 'CAR 3:0', 'CAR 4:0', 'CAR 4:0;O', 'CAR 5:0', 'CAR 16:0', 'CAR 18:0', 'CAR 18:1', 'CAR 20:1', 'CAR 20:4', 'NAOrn 19:0;O', 'NAOrn 19:0;O']
data_Ar = data.loc[(data['Name'] == name_list[0]) | (data['Name'] == name_list[1]) | (data['Name'] == name_list[2]) | (data['Name'] == name_list[3]) | (data['Name'] == name_list[4]) | 
                   (data['Name'] == name_list[5]) | (data['Name'] == name_list[6]) | (data['Name'] == name_list[7]) | (data['Name'] == name_list[8]) | (data['Name'] == name_list[9]) | 
                   (data['Name'] == name_list[10]) | (data['Name'] == name_list[11])].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)]
data_Br = data_Ar
polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)

# Bases
name_list = list(data.loc[((data['Class'] == 'Base') | (data['Class'] == 'Phosphate')) & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Name'])
des_list = list(data.loc[((data['Class'] == 'Base') | (data['Class'] == 'Phosphate')) & ((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)), 'Class'])
plot_gene_group_select_tissue_new(data, name_list, 'Rhythmic Base in E19 SCN', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)
plot_gene_group_select_tissue_review(data, name_list, 'Rhythmic Base in E19 PLS', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=20, ano2w=False)

data_Ar = data.loc[(data['Name'] == name_list[0]) | (data['Name'] == name_list[1]) | (data['Name'] == name_list[2]) | (data['Name'] == name_list[3]) | 
                   (data['Name'] == name_list[4]) | (data['Name'] == name_list[5])].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)]
data_Br = data_Ar
polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)


# inspired by https://www.cell.com/cell/pdf/S0092-8674(18)31103-6.pdf
# calculate rho Spearman coeff for C v S and plot as correlation heatmap or hinton plot
# Correlation matrix visualized by Hinton plot
# http://matplotlib.org/examples/specialty_plots/hinton_demo.html
from matplotlib.collections import PolyCollection
def hinton_poly(matrix, max_weight=None, ax=None, positive_cor='red', negative_cor='blue', background='lightgray', rasterized=False, nth=10, use_ticks=True):  
    #Draw Hinton diagram for visualizing a weight matrix.
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

    ax.patch.set_facecolor(background)   # 'lightgray'
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # Create a list to store the vertices of each rectangle
    rects = []

    for (x, y), w in np.ndenumerate(matrix):
        # color = positive_cor if w > 0 else negative_cor    # color = 'red' if w > 0 else 'blue'
        size = np.sqrt(np.abs(w))
        verts = [
            (x - size / 2, y - size / 2),
            (x + size / 2, y - size / 2),
            (x + size / 2, y + size / 2),
            (x - size / 2, y + size / 2)
        ]
        rects.append(verts)

    # Create a PolyCollection object with the rectangles
    # poly_collection = PolyCollection(rects, facecolors=[positive_cor if w > 0 else negative_cor for w in matrix.flatten()], edgecolors=[positive_cor if w > 0 else negative_cor for w in matrix.flatten()], rasterized=rasterized)
    poly_collection = PolyCollection(rects, facecolors=[positive_cor if w > 0 else negative_cor for w in matrix.to_numpy().flatten()], 
                                     edgecolors=[positive_cor if w > 0 else negative_cor for w in matrix.to_numpy().flatten()], rasterized=rasterized)

    # Add the PolyCollection to the axis
    ax.add_collection(poly_collection)
    
    if use_ticks == False:
        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        
        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
    
    else:        
        nticks = matrix.shape[0]
        ax.set_xticks(range(nticks))
        
        temp = ax.xaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::nth]))
        for label in temp:
            label.set_visible(False)    
        
        ax.set_xticklabels(list(matrix.columns), fontsize='xx-small', rotation=90)
        ax.set_yticks(range(nticks))
        temp = ax.yaxis.get_ticklabels()
        temp = list(set(temp) - set(temp[::nth]))
        for label in temp:
            label.set_visible(False)    
        
        ax.set_yticklabels(matrix.columns, fontsize='xx-small')
        
    ax.grid(False)
    ax.autoscale_view()
    ax.invert_yaxis()
    # plt.show()

# create correlations between all variables like this, alt method='pearson', uses FAST VERSION of hinton_poly
def Correlation2(data, columns, name, mydir, method='spearman', positive_cor='red', negative_cor='blue', background='lightgray', rasterized=False, nth=10):
    dta = data[columns]    
    corrmat = dta.corr(method=method)   # pandas method .corr()
    #sns.heatmap(corrmat, vmax=1., square=False)    #.xaxis.tick_top()  # if heatmap is preferable than hinton
    hinton_poly(corrmat, positive_cor=positive_cor, negative_cor=negative_cor, background=background, rasterized=rasterized, nth=nth)
    plt.savefig(f'{mydir}\\Correlation_Hinton_{name}.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\Correlation_Hinton_{name}.svg', format = 'svg', bbox_inches = 'tight', dpi=600)
    plt.clf()
    plt.close()


# reload data if necessary, backup
# data = pd.read_csv('_data_all_with_calculations.csv')
# data.drop(columns='Unnamed: 0', inplace=True)

# Sort data by value counts in Class and correlate means only across times
cordat = data[['Class', 'Name','E19_00',
'E19_04', 'E19_08', 'E19_12', 'E19_16', 'E19_20', 'E19_24',
'P02_00', 'P02_04', 'P02_08', 'P02_12', 'P02_16', 'P02_20',
'P02_24', 'P10_00', 'P10_04', 'P10_08', 'P10_12', 'P10_16',
'P10_20', 'P10_24', 'P20_00', 'P20_04', 'P20_08', 'P20_12',
'P20_16', 'P20_20', 'P20_24', 'P28_00', 'P28_04', 'P28_08',
'P28_12', 'P28_16', 'P28_20', 'P28_24']]
# Calculate value counts for the 'Class' column
class_counts = cordat['Class'].value_counts()
# Sort the DataFrame by the counts of values in the 'Class' column
sorted_data = cordat.merge(class_counts.rename('Count'), left_on='Class', right_index=True)
sorted_data.sort_values(by='Count', ascending=False, inplace=True)
sorted_data.drop(columns='Count', inplace=True)
print(sorted_data)



# Hinton plot

# E19 group
E19_cor = sorted_data.loc[:, 'E19_00':"E19_24"].T.astype(float)
rhovalues_E19 = E19_cor.astype(float).corr(method='spearman')
# P02 group
P02_cor = sorted_data.loc[:, 'P02_00':"P02_24"].T.astype(float)
rhovalues_P02 = P02_cor.astype(float).corr(method='spearman')
# P10 group
P10_cor = sorted_data.loc[:, 'P10_00':"P10_24"].T.astype(float)
rhovalues_P10 = P10_cor.astype(float).corr(method='spearman')
# P20 group
P20_cor = sorted_data.loc[:, 'P20_00':"P20_24"].T.astype(float)
rhovalues_P20 = P20_cor.astype(float).corr(method='spearman')
# P28 group
P28_cor = sorted_data.loc[:, 'P28_00':"P28_24"].T.astype(float)
rhovalues_P28 = P28_cor.astype(float).corr(method='spearman')

# To plot only major classes without mRNA and small classes
sorted_data_m = sorted_data.iloc[0:799, :]

# E19 group
E19_cor = sorted_data_m.loc[:, 'E19_00':"E19_24"].T.astype(float)
rhovalues_E19 = E19_cor.astype(float).corr(method='spearman')
# P02 group
P02_cor = sorted_data_m.loc[:, 'P02_00':"P02_24"].T.astype(float)
rhovalues_P02 = P02_cor.astype(float).corr(method='spearman')
# P10 group
P10_cor = sorted_data_m.loc[:, 'P10_00':"P10_24"].T.astype(float)
rhovalues_P10 = P10_cor.astype(float).corr(method='spearman')
# P20 group
P20_cor = sorted_data_m.loc[:, 'P20_00':"P20_24"].T.astype(float)
rhovalues_P20 = P20_cor.astype(float).corr(method='spearman')
# P28 group
P28_cor = sorted_data_m.loc[:, 'P28_00':"P28_24"].T.astype(float)
rhovalues_P28 = P28_cor.astype(float).corr(method='spearman')


# calculate p values
from scipy.stats import spearmanr

# this computes the p-values  VERY SLOW, to avoid read csv of previously calculated values, uncomment if first run
# pvalues_E19 = E19_cor.astype(float).corr(method=lambda x, y: spearmanr(x, y)[1]) # - np.eye(len(df.columns)) 
# pvalues_E19.to_csv('Hinton_pvalues_E19_m.csv')
# pvalues_E19 = pd.read_csv('Hinton_pvalues_E19.csv', index_col='Unnamed: 0')
pvalues_E19 = pd.read_csv('Hinton_pvalues_E19_m.csv', index_col='Unnamed: 0') # only major classes
# count values < 0.05
pvalues_E19n = (pvalues_E19 < 0.05).sum().sum()
rhovalues_E19plus = (rhovalues_E19 < 0).sum().sum()
rhovalues_E19minus = ((rhovalues_E19 > 0) & (rhovalues_E19 < 1)).sum().sum()
pvalues_E19nPlus = ((pvalues_E19 < 0.05) & (rhovalues_E19 > 0) & (rhovalues_E19 < 1)).sum().sum()
pvalues_E19nMinus = ((pvalues_E19 < 0.05) & (rhovalues_E19 < 0)).sum().sum()

# this computes the p-values  VERY SLOW, to avoid read csv of previously calculated values, uncomment if first run
# pvalues_P02 = P02_cor.astype(float).corr(method=lambda x, y: spearmanr(x, y)[1]) # - np.eye(len(df.columns)) 
# pvalues_P02.to_csv('Hinton_pvalues_P02_m.csv')
# pvalues_P02 = pd.read_csv('Hinton_pvalues_P02.csv', index_col='Unnamed: 0')
pvalues_P02 = pd.read_csv('Hinton_pvalues_P02_m.csv', index_col='Unnamed: 0')
pvalues_P02n = (pvalues_P02 < 0.05).sum().sum()
rhovalues_P02plus = (rhovalues_P02 < 0).sum().sum()
rhovalues_P02minus = ((rhovalues_P02 > 0) & (rhovalues_P02 < 1)).sum().sum()
pvalues_P02nPlus = ((pvalues_P02 < 0.05) & (rhovalues_P02 > 0) & (rhovalues_P02 < 1)).sum().sum()
pvalues_P02nMinus = ((pvalues_P02 < 0.05) & (rhovalues_P02 < 0)).sum().sum()

# this computes the p-values  VERY SLOW, to avoid read csv of previously calculated values, uncomment if first run
# pvalues_P10 = P10_cor.astype(float).corr(method=lambda x, y: spearmanr(x, y)[1]) # - np.eye(len(df.columns)) 
# pvalues_P10.to_csv('Hinton_pvalues_P10_m.csv')
# pvalues_P10 = pd.read_csv('Hinton_pvalues_P10.csv', index_col='Unnamed: 0')
pvalues_P10 = pd.read_csv('Hinton_pvalues_P10_m.csv', index_col='Unnamed: 0')
pvalues_P10n = (pvalues_P10 < 0.05).sum().sum()
rhovalues_P10plus = (rhovalues_P10 < 0).sum().sum()
rhovalues_P10minus = ((rhovalues_P10 > 0) & (rhovalues_P10 < 1)).sum().sum()
pvalues_P10nPlus = ((pvalues_P10 < 0.05) & (rhovalues_P10 > 0) & (rhovalues_P10 < 1)).sum().sum()
pvalues_P10nMinus = ((pvalues_P10 < 0.05) & (rhovalues_P10 < 0)).sum().sum()

# this computes the p-values  VERY SLOW, to avoid read csv of previously calculated values, uncomment if first run
# pvalues_P20 = P20_cor.astype(float).corr(method=lambda x, y: spearmanr(x, y)[1]) # - np.eye(len(df.columns)) 
# pvalues_P20.to_csv('Hinton_pvalues_P20_m.csv')
# pvalues_P20 = pd.read_csv('Hinton_pvalues_P20.csv', index_col='Unnamed: 0')
pvalues_P20 = pd.read_csv('Hinton_pvalues_P20_m.csv', index_col='Unnamed: 0')
pvalues_P20n = (pvalues_P20 < 0.05).sum().sum()
rhovalues_P20plus = (rhovalues_P20 < 0).sum().sum()
rhovalues_P20minus = ((rhovalues_P20 > 0) & (rhovalues_P20 < 1)).sum().sum()
pvalues_P20nPlus = ((pvalues_P20 < 0.05) & (rhovalues_P20 > 0) & (rhovalues_P20 < 1)).sum().sum()
pvalues_P20nMinus = ((pvalues_P20 < 0.05) & (rhovalues_P20 < 0)).sum().sum()

# this computes the p-values  VERY SLOW, to avoid read csv of previously calculated values, uncomment if first run
# pvalues_P28 = P28_cor.astype(float).corr(method=lambda x, y: spearmanr(x, y)[1]) # - np.eye(len(df.columns)) 
# pvalues_P28.to_csv('Hinton_pvalues_P28_m.csv')
# pvalues_P28 = pd.read_csv('Hinton_pvalues_P28.csv', index_col='Unnamed: 0')
pvalues_P28 = pd.read_csv('Hinton_pvalues_P28_m.csv', index_col='Unnamed: 0')
pvalues_P28n = (pvalues_P28 < 0.05).sum().sum()
rhovalues_P28plus = (rhovalues_P28 < 0).sum().sum()
rhovalues_P28minus = ((rhovalues_P28 > 0) & (rhovalues_P28 < 1)).sum().sum()
pvalues_P28nPlus = ((pvalues_P28 < 0.05) & (rhovalues_P28 > 0) & (rhovalues_P28 < 1)).sum().sum()
pvalues_P28nMinus = ((pvalues_P28 < 0.05) & (rhovalues_P28 < 0)).sum().sum()


# Barplot of total n of correlations with p > 0.05
# total values
n = 856 * 856
n = 799 * 799 # for sorted_data_m dataset with major classes only
# Plot n of correlations with stacked barplot
fig, ax = plt.subplots(figsize=(4, 8))
grps = ('E19', 'P02', 'P10', 'P20', 'P28')
s_counts = {
    'Positive': [pvalues_E19nPlus, pvalues_P02nPlus, pvalues_P10nPlus, pvalues_P20nPlus, pvalues_P28nPlus],
    'Negative': [pvalues_E19nMinus, pvalues_P02nMinus, pvalues_P10nMinus, pvalues_P20nMinus, pvalues_P28nMinus]
}
width = 0.6  # the width of the bars: can also be len(x) sequence
bottom = np.zeros(5)
for sex, s_count in s_counts.items():
    p = ax.bar(grps, s_count, width, label=sex, bottom=bottom) # 
    bottom += s_count
    ax.bar_label(p, label_type='center')
ax.set_title('Number of significant metabolite correlations')
ax.legend()
plt.savefig(f'{mydir}\\Correlation_barplot_m.png', format = 'png', bbox_inches = 'tight')
plt.savefig(f'{mydir}\\Correlation_barplot_m.svg', format = 'svg', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()

# function for Fig. 2B
def Correlation4(data, mask, name, mydir, method='spearman', positive_cor='red', negative_cor='blue', background='lightgray', rasterized=False, nth=10, use_ticks=True): 
    corrmat = data.corr(method=method)  # pandas method .corr()
    #sns.heatmap(corrmat[mask], vmax=1., square=False, cmap=sns.cubehelix_palette(as_cmap=True))    #.xaxis.tick_top()  # if heatmap is preferable than hinton
    hinton_poly(corrmat[mask], positive_cor=positive_cor, negative_cor=negative_cor, background=background, rasterized=rasterized, nth=nth, use_ticks=use_ticks)
    plt.savefig(f'{mydir}\\Correlation_Hinton_{name}_masked.png', format = 'png', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\Correlation_Hinton_{name}_masked.svg', format = 'svg', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\Correlation_Hinton_{name}_masked.eps', format = 'eps', bbox_inches = 'tight')
    plt.clf()
    plt.close()

### To save figs as vector svg with fonts editable in Corel ###
import matplotlib as mpl
# mpl.use('svg') # mpl.use('Qt5Agg') #to turn interactive backend on
# ['agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'] ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)
    
# Fig. 2B
Correlation4(E19_cor.astype(float), (pvalues_E19 < 0.0001) & ((rhovalues_E19 >= 0.9) | (rhovalues_E19 <= -0.9)), 'E19', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P02_cor.astype(float), (pvalues_P02 < 0.0001) & ((rhovalues_P02 >= 0.9) | (rhovalues_P02 <= -0.9)), 'P02', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P10_cor.astype(float), (pvalues_P10 < 0.0001) & ((rhovalues_P10 >= 0.9) | (rhovalues_P10 <= -0.9)), 'P10', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P20_cor.astype(float), (pvalues_P20 < 0.0001) & ((rhovalues_P20 >= 0.9) | (rhovalues_P20 <= -0.9)), 'P20', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P28_cor.astype(float), (pvalues_P28 < 0.0001) & ((rhovalues_P28 >= 0.9) | (rhovalues_P28 <= -0.9)), 'P28', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)

Correlation4(E19_cor.astype(float), (pvalues_E19 < 0.05) & ((rhovalues_E19 >= 0.5) | (rhovalues_E19 <= -0.5)), 'E19 v3', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P02_cor.astype(float), (pvalues_P02 < 0.05) & ((rhovalues_P02 >= 0.5) | (rhovalues_P02 <= -0.5)), 'P02 v3', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P10_cor.astype(float), (pvalues_P10 < 0.05) & ((rhovalues_P10 >= 0.5) | (rhovalues_P10 <= -0.5)), 'P10 v3', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P20_cor.astype(float), (pvalues_P20 < 0.05) & ((rhovalues_P20 >= 0.5) | (rhovalues_P20 <= -0.5)), 'P20 v3', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)
Correlation4(P28_cor.astype(float), (pvalues_P28 < 0.05) & ((rhovalues_P28 >= 0.5) | (rhovalues_P28 <= -0.5)), 'P28 v3', mydir, method='spearman', positive_cor='tomato', negative_cor='slateblue', background='white', rasterized=True, nth=50, use_ticks=False)


# Follow up for Hinton plot> in corel or inkscape add these color bands as legend for hinton
import matplotlib.patches as mpatches
# Get the counts of each class
# class_counts = sorted_data['Class'].value_counts()
class_counts = sorted_data_m['Class'].value_counts()
# Create a new dataframe with counts
df_counts = pd.DataFrame(class_counts).reset_index()
df_counts.columns = ['Class', 'Count']
# Create a color map for different classes using default Seaborn palette
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
gist = mpl.colormaps['gist_ncar']
tab20 = mpl.colormaps['tab20']
colors = sns.color_palette()[:len(df_counts)]
# add more colors manually, otherwise comment out
clist = tab20(range(18))
for x in clist[0:18]:
    colors.append(x)
# Initialize the plot
fig, ax = plt.subplots()
# Calculate the total count
total_count = df_counts['Count'].sum()
# Initialize the y position
y_position = 0
# Plot each segment and create legend entries
legend_handles = []
for clso, count, color in zip(df_counts['Class'], df_counts['Count'], colors):
    bar = ax.bar(0, count, bottom=y_position, color=color)
    legend_handles.append(mpatches.Patch(color=color, label=clso))
    y_position += count
# Add labels and title
plt.xlabel('Class')
plt.ylabel('Count')
# Set the x-axis limits
ax.set_xlim(-0.5, 0.5)
# Add legend with custom handles
plt.legend(handles=legend_handles, loc='upper right')
# Hide the x-axis
ax.xaxis.set_visible(False)
# Show the plot
plt.savefig(f'{mydir}\\Correlation_legend v3.png', format = 'png', bbox_inches = 'tight')
plt.savefig(f'{mydir}\\Correlation_legend v3.svg', format = 'svg', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()



# EXPORT DATA FOR CHORD DIAGRAM with P-VAL FILTER

# keep only lower triangle of identical values in scatter matrix of correlations
rhovalues_E19_Class = rhovalues_E19.mask(np.tril(np.ones(rhovalues_E19.shape)).astype(bool))
rhovalues_E19_Class[(rhovalues_E19_Class < 0.9) & (rhovalues_E19_Class > -0.9)] = 0
rhovalues_E19_Class[(rhovalues_E19_Class >= 0.9) & (pvalues_E19 < 0.001)] = 1
rhovalues_E19_Class[(rhovalues_E19_Class >= 0.9) & (pvalues_E19 >= 0.001)] = 0
rhovalues_E19_Class[(rhovalues_E19_Class <= -0.9)] = 0
rhovalues_E19_Class.columns = sorted_data_m['Class']
rhovalues_E19_Class.index = sorted_data_m['Class']
# need to also change 1 where the same metabolite correlates with itself
for i in range(len(rhovalues_E19_Class.columns)):
    rhovalues_E19_Class.iloc[i, i] = 0
# Sum the rows by their respective labels
summed_rows = rhovalues_E19_Class.groupby(rhovalues_E19_Class.index).sum()
# Sum the columns by their respective labels
summed_columns = summed_rows.groupby(summed_rows.columns, axis=1).sum()
summed_columns = summed_columns.astype(int)
total_Cp = summed_columns.sum().sum()
# optional hack for sparse chord plot with more stringent mas (e.g. p<0.001)
summed_columns.loc[summed_columns.sum() == 0, summed_columns.sum() == 0] = 1
summed_columns.to_csv('_chordtable_rhovalues_sum_E19_positive.csv')

# keep only lower triangle of identical values in scatter matrix of correlations
rhovalues_P02_Class = rhovalues_P02.mask(np.tril(np.ones(rhovalues_P02.shape)).astype(bool))
rhovalues_P02_Class[(rhovalues_P02_Class < 0.9) & (rhovalues_P02_Class > -0.9)] = 0
rhovalues_P02_Class[(rhovalues_P02_Class >= 0.9) & (pvalues_P02 < 0.001)] = 1
rhovalues_P02_Class[(rhovalues_P02_Class >= 0.9) & (pvalues_P02 >= 0.001)] = 0
rhovalues_P02_Class[(rhovalues_P02_Class <= -0.9)] = 0
rhovalues_P02_Class.columns = sorted_data_m['Class']
rhovalues_P02_Class.index = sorted_data_m['Class']
# need to also change 1 where the same metabolite correlates with itself
for i in range(len(rhovalues_P02_Class.columns)):
    rhovalues_P02_Class.iloc[i, i] = 0
# Sum the rows by their respective labels
summed_rows = rhovalues_P02_Class.groupby(rhovalues_P02_Class.index).sum()
# Sum the columns by their respective labels
summed_columns = summed_rows.groupby(summed_rows.columns, axis=1).sum()
summed_columns = summed_columns.astype(int)
total_Cp = summed_columns.sum().sum()
# optional hack for sparse chord plot with more stringent mas (e.g. p<0.001)
summed_columns.loc[summed_columns.sum() == 0, summed_columns.sum() == 0] = 1
summed_columns.to_csv('_chordtable_rhovalues_sum_P02_positive.csv')

# keep only lower triangle of identical values in scatter matrix of correlations
rhovalues_P10_Class = rhovalues_P10.mask(np.tril(np.ones(rhovalues_P10.shape)).astype(bool))
rhovalues_P10_Class[(rhovalues_P10_Class < 0.9) & (rhovalues_P10_Class > -0.9)] = 0
rhovalues_P10_Class[(rhovalues_P10_Class >= 0.9) & (pvalues_P10 < 0.001)] = 1
rhovalues_P10_Class[(rhovalues_P10_Class >= 0.9) & (pvalues_P10 >= 0.001)] = 0
rhovalues_P10_Class[(rhovalues_P10_Class <= -0.9)] = 0
rhovalues_P10_Class.columns = sorted_data_m['Class']
rhovalues_P10_Class.index = sorted_data_m['Class']
# need to also change 1 where the same metabolite correlates with itself
for i in range(len(rhovalues_P10_Class.columns)):
    rhovalues_P10_Class.iloc[i, i] = 0
# Sum the rows by their respective labels
summed_rows = rhovalues_P10_Class.groupby(rhovalues_P10_Class.index).sum()
# Sum the columns by their respective labels
summed_columns = summed_rows.groupby(summed_rows.columns, axis=1).sum()
summed_columns = summed_columns.astype(int)
total_Cp = summed_columns.sum().sum()
# optional hack for sparse chord plot with more stringent mas (e.g. p<0.001)
summed_columns.loc[summed_columns.sum() == 0, summed_columns.sum() == 0] = 1
summed_columns.to_csv('_chordtable_rhovalues_sum_P10_positive.csv')

# keep only lower triangle of identical values in scatter matrix of correlations
rhovalues_P20_Class = rhovalues_P20.mask(np.tril(np.ones(rhovalues_P20.shape)).astype(bool))
rhovalues_P20_Class[(rhovalues_P20_Class < 0.9) & (rhovalues_P20_Class > -0.9)] = 0
rhovalues_P20_Class[(rhovalues_P20_Class >= 0.9) & (pvalues_P20 < 0.001)] = 1
rhovalues_P20_Class[(rhovalues_P20_Class >= 0.9) & (pvalues_P20 >= 0.001)] = 0
rhovalues_P20_Class[(rhovalues_P20_Class <= -0.9)] = 0
rhovalues_P20_Class.columns = sorted_data_m['Class']
rhovalues_P20_Class.index = sorted_data_m['Class']
# need to also change 1 where the same metabolite correlates with itself
for i in range(len(rhovalues_P20_Class.columns)):
    rhovalues_P20_Class.iloc[i, i] = 0
# Sum the rows by their respective labels
summed_rows = rhovalues_P20_Class.groupby(rhovalues_P20_Class.index).sum()
# Sum the columns by their respective labels
summed_columns = summed_rows.groupby(summed_rows.columns, axis=1).sum()
summed_columns = summed_columns.astype(int)
total_Cp = summed_columns.sum().sum()
# optional hack for sparse chord plot with more stringent mas (e.g. p<0.001)
summed_columns.loc[summed_columns.sum() == 0, summed_columns.sum() == 0] = 1
summed_columns.to_csv('_chordtable_rhovalues_sum_P20_positive.csv')

# keep only lower triangle of identical values in scatter matrix of correlations
rhovalues_P28_Class = rhovalues_P28.mask(np.tril(np.ones(rhovalues_P28.shape)).astype(bool))
rhovalues_P28_Class[(rhovalues_P28_Class < 0.9) & (rhovalues_P28_Class > -0.9)] = 0
rhovalues_P28_Class[(rhovalues_P28_Class >= 0.9) & (pvalues_P28 < 0.001)] = 1
rhovalues_P28_Class[(rhovalues_P28_Class >= 0.9) & (pvalues_P28 >= 0.001)] = 0
rhovalues_P28_Class[(rhovalues_P28_Class <= -0.9)] = 0
rhovalues_P28_Class.columns = sorted_data_m['Class']
rhovalues_P28_Class.index = sorted_data_m['Class']
# need to also change 1 where the same metabolite correlates with itself
for i in range(len(rhovalues_P28_Class.columns)):
    rhovalues_P28_Class.iloc[i, i] = 0
# Sum the rows by their respective labels
summed_rows = rhovalues_P28_Class.groupby(rhovalues_P28_Class.index).sum()
# Sum the columns by their respective labels
summed_columns = summed_rows.groupby(summed_rows.columns, axis=1).sum()
summed_columns = summed_columns.astype(int)
total_Cp = summed_columns.sum().sum()
# optional hack for sparse chord plot with more stringent mas (e.g. p<0.001)
summed_columns.loc[summed_columns.sum() == 0, summed_columns.sum() == 0] = 1
summed_columns.to_csv('_chordtable_rhovalues_sum_P28_positive.csv')


# functions for Chord diagram inspired by this:
# https://moshi4.github.io/pyCirclize/chord_diagram/
# https://jokergoo.github.io/circlize_book/book/the-chorddiagram-function.html
# https://medium.com/@sy.41211/pycirclize-circular-visualization-in-python-b742ec5d45ac

# MAY NEED TO RUN in separate env due to
# installation issue with pyCirclize > conda activate graf

from pycirclize import Circos
from pycirclize.parser import Matrix
from pycirclize.utils import ColorCycler
from matplotlib.patches import Patch

# Function to map values to colors
def value_to_color(value):
    if value > 0:
        return 'red'
    else:
        return 'blue'

# Define link_kws handler function to customize each link property by Class
def link_kws_handler(from_label: str, to_label: str):
    if from_label in ("Polar", "CholesterolSulfate") and to_label in ("TAG"):
        # Set alpha, zorder values higher than other links for highlighting
        return dict(alpha=0.8, zorder=1.0)
    else:
        return dict(alpha=0.1, zorder=0)


# Customizing Circos plot
# space = 3 --- sets space between Classes rectangles
# r_lim --- sets width of Classes rectangles

# Load all data used in plots that should be compared, BUG - no column may be entirely 0 - edit data0 to add 1x 1 to cholesterol in case of p<0.0001
matrix_data0 = pd.read_csv('_chordtable_rhovalues_sum_E19_positive.csv', delimiter = ',', encoding = "utf-8", low_memory=False, index_col = 'Class')
matrix_data1 = pd.read_csv('_chordtable_rhovalues_sum_P02_positive.csv', delimiter = ',', encoding = "utf-8", low_memory=False, index_col = 'Class')
matrix_data2 = pd.read_csv('_chordtable_rhovalues_sum_P10_positive.csv', delimiter = ',', encoding = "utf-8", low_memory=False, index_col = 'Class')
matrix_data3 = pd.read_csv('_chordtable_rhovalues_sum_P20_positive.csv', delimiter = ',', encoding = "utf-8", low_memory=False, index_col = 'Class')
matrix_data4 = pd.read_csv('_chordtable_rhovalues_sum_P28_positive.csv', delimiter = ',', encoding = "utf-8", low_memory=False, index_col = 'Class')


# Calculate gaps/spaces to scale the plots to the same size, then multiply spaces by this number
s0 = matrix_data0.sum().sum()
s1 = matrix_data1.sum().sum()
s2 = matrix_data2.sum().sum()
s3 = matrix_data3.sum().sum()
s4 = matrix_data4.sum().sum()
slist = [s0, s1, s2, s3, s4]
spaces = []
for s in range(len(slist)):
    if slist[s] < max(slist):
        spaces.append(round(max(slist)/slist[s], 1))
    else:
        spaces.append(1)       


import matplotlib as mpl
gist = mpl.colormaps['gist_ncar']
tab20 = mpl.colormaps['tab20']
tab20b = mpl.colormaps['tab20b']
# colors = np.append(tab20(range(20)), tab20b(range(20)))

# colors in total
colors1 = plt.cm.gist_heat_r(np.linspace(0., 1, 128))
colors2 = plt.cm.gist_ncar(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = LinearSegmentedColormap.from_list('my_colormap', colors)

# Fig. 2D
circos = Circos.initialize_from_matrix(matrix_data3, space=int(round(3*spaces[0], 1)), r_lim=(93, 100), ticks_interval=1000, label_kws=dict(r=94, size=10, color="black"), link_kws=dict(linewidth=0), cmap='tab20')
fig = circos.plotfig()
rect_handles = []
rect_colors = ColorCycler.get_color_list()
# rect_colors = rect_colors[1::9][0:28] get evry 9nth and make it 28 long
for idx, color in enumerate(rect_colors, 1):
    rect_handles.append(Patch(color=color, label=f"{matrix_data0.columns[idx]}"))
_ = circos.ax.legend(handles=rect_handles, bbox_to_anchor=(0.1, 0.1),
    fontsize=8,
    title="Class",
    ncol=2,
    loc="upper right")
fig.savefig("_chord_diagram_rhovalues_sum_E19_positive_legend.png")
fig.savefig('_chord_diagram_rhovalues_sum_E19_positive_legend.svg')


# Fig. 2A
# for PCA, use processed data - z-scores from all ages and all tissues together
df = data.loc[:, 'score_E19_ZT00_1':'score_P28_ZT24_5']
# df.insert(0, 'Name', data['Name'])
dfn = df.loc[:, df.columns[0:]]
data_melted = dfn.melt(var_name="Timepoint", value_name="Value")
data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')
data_melted["Time"] = pd.to_numeric(data_melted["Time"])
data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])
# take transposed data with analytes in columns and group/time as well, pick only z-scored values
dta = data_T.iloc[0:175, :]
# PCT needs analytes in rows for some reason, dta.T
pca_model = PCA(dta.iloc[:, 3:].dropna().T, standardize=False, demean=True)
# but PCA plots need analytes in columns again with hue in columns as well
PCA_plot(dta, dta.columns[3:], hue = 'Group', name = 'Group', mydir = mydir, pca_model=pca_model, pc='PC1')


# Fig. 2E and other volcanos
# calculate deseq2 in RNAlysis, load and merge
E19_P28_deseq2 = pd.read_csv('.\RNAlysis\deseq2\DESeq2_Group_E19_vs_P28 with0.csv').rename(columns={'Unnamed: 0': 'Original_row', 'baseMean': 'E19vsP28_baseMean', 'log2FoldChange': 'E19vsP28_log2FoldChange', 
                                                                                                    'lfcSE': 'E19vsP28_lfcSE', 'stat': 'E19vsP28_stat', 'pvalue': 'E19vsP28_pvalue','padj': 'E19vsP28_padj'})

data = data.merge(E19_P28_deseq2, how='outer')

# volcano of all significant analytes, including those with 0 in some age -def volcano_annotated(data, logFC_col, pval_col, mydir, logFC_cutoff=2, pval_cutoff=2, save_list = True):
volcano_annotated(data, logFC_col='E19vsP28_log2FoldChange', pval_col='E19vsP28_padj', mydir = mydir, logFC_cutoff=2, pval_cutoff=2, save_list = True)  

# to only show analytes without 0, drop them like this
col_names=data.loc[:,'raw_E19_ZT00_1':'raw_P28_ZT24_5'].columns  # even combining>>> col_names=df.loc[:,'raw_E19_ZT00_1':'raw_P28_ZT24_5'].columns + df.loc[:,'col120':'col220'].columns
volcano_annotated(data.dropna(subset = col_names), logFC_col='E19vsP28_log2FoldChange', mydir = mydir, pval_col='E19vsP28_padj', logFC_cutoff=2, pval_cutoff=2, save_list = False)  


# manually created file with all non0 analytes and at which onto age (only combinations making sense) they are detected
when_detected = pd.read_csv('when_detected.csv')
# check for errors 
# when_detected[when_detected.duplicated(['Original_row'])]
# merge
data = data.merge(when_detected, how='outer')
data['detected'].value_counts()

# final complete datafile
data.to_csv('_data_all_with_calculations.csv')



# Fig. S3
# plot traces of specific analytes
name_list = list(data.loc[data['detected'] == 'E19', 'Name'])
des_list = list(data.loc[data['detected'] == 'E19', 'KEGG']) # use Class or KEGG for des.
# plot_gene_group(data, name_list, 'prenatal - only E19', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=40, ano2w=False)
plot_gene_group_select_tissue_new(data, name_list, 'prenatal - only E19 v2', tissue='SCN', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['detected'] == 'E19+P02', 'Name'])
des_list = list(data.loc[data['detected'] == 'E19+P02', 'KEGG'])
plot_gene_group_select_tissue_new(data, name_list, 'perinatal - only E19+P02 v2', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['detected'] == 'P20+P28', 'Name'])
des_list = list(data.loc[data['detected'] == 'P20+P28', 'KEGG'])
plot_gene_group_select_tissue_new(data, name_list, 'weaning - P20+P28 v2', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['detected'] == 'P02+P10+P20+P28', 'Name'])
des_list = list(data.loc[data['detected'] == 'P02+P10+P20+P28', 'KEGG'])
plot_gene_group_select_tissue_new(data, name_list, 'postnatal - P02+P10+P20+P28 v2', tissue='SCN', annotation=False, describe=True, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['detected'] == 'P28', 'Name'])
des_list = list(data.loc[data['detected'] == 'P28', 'KEGG'])
plot_gene_group_select_tissue_new(data, name_list, 'after weaning - P28 v2', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=40, ano2w=False)

name_list = list(data.loc[data['detected'] == 'E19+P02+P10+P20', 'Name'])
des_list = list(data.loc[data['detected'] == 'E19+P02+P10+P20', 'KEGG'])
plot_gene_group_select_tissue_new(data, name_list, 'until weaning - E19+P02+P10+P20 v2', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=40, ano2w=False)


# Fig. S1
# plot traces after deseq2, only those in all ages that are dif.. between E19 and P28
pval_cutoff=2
logFC_cutoff=2
logFC_col='E19vsP28_log2FoldChange'
pval_col='E19vsP28_padj'
col_names=data.loc[:,'raw_E19_ZT00_1':'raw_P28_ZT24_5'].columns
uplist = data.dropna(subset = col_names)[(data[logFC_col]<=- logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]
name_list = list(uplist['Name'])
des_list = name_list
plot_gene_group_select_tissue_new(data, name_list, 'E19 vs P28 low level', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=False, fnt=40, ano2w=False)
# Fig. S2
downlist = data.dropna(subset = col_names)[(data[logFC_col]>= logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]
name_list = list(downlist['Name'])
des_list = name_list
# plot_gene_group(data, name_list, 'E19 vs P28 v4', annotation=False, describe=False, des_list=des_list, y_norm=False, fnt=40, ano2w=False)
plot_gene_group_select_tissue_new(data, name_list, 'E19 vs P28 v4', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=False, fnt=40, ano2w=False)

# include NaNs, parts to Fig. 2F
pval_cutoff=2
logFC_cutoff=2
logFC_col='E19vsP28_log2FoldChange'
pval_col='E19vsP28_padj'
uplist = data[(data[logFC_col]<=- logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]
name_list = list(uplist['Name'])
des_list = name_list
plot_gene_group_select_tissue_new(data, name_list, 'E19 vs P28 increasing updated', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=False, fnt=80, ano2w=False)
# parts to Fig. 2G
downlist = data[(data[logFC_col]>= logFC_cutoff)&(data[pval_col]<= pval_cutoff)][['Name', 'Metabolite name (LORA format)', 'HMDB', 'KEGG', 'PubChem', pval_col]]
name_list = list(downlist['Name'])
des_list = name_list
plot_gene_group_select_tissue_new(data, name_list, 'E19 vs P28 decreasing updated', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=False, fnt=80, ano2w=False)


# plot individual analytes
plot_name_ages(data, "Spermidine", y_norm=False, errorbar='sem', log=True)

# in PLS
plot_name_ages_select_tissue(data, "Thymidine", tissue='PLS', y_norm=False, errorbar='sem', log=True)


# Export for ORA and others
# background
background_set = data.iloc[:-5, :].loc[:, 'Metabolite name (LORA format)'].to_csv('ora_background.csv', index=False)
background_set = data.iloc[:-5, :].loc[:, 'KEGG'].dropna().to_csv('ora_background KEGG.csv', index=False)
background_set = data.iloc[:-5, :].loc[:, 'PubChem'].dropna().to_csv('ora_background PubChem.csv', index=False)
background_set = data.iloc[:-5, :].loc[:, 'HMDB'].dropna().to_csv('ora_background HMDBm.csv', index=False)

#rhythmic
data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'Metabolite name (LORA format)'].to_csv('ora_rhythmic_E19.csv', index=False)
data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Metabolite name (LORA format)'].to_csv('ora_rhythmic_P28.csv', index=False)

data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'KEGG'].dropna().to_csv('ora_rhythmic_E19 KEGG.csv', index=False)
data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'KEGG'].dropna().to_csv('ora_rhythmic_P28 KEGG.csv', index=False)

data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'PubChem'].dropna().to_csv('ora_rhythmic_E19 PubChem.csv', index=False)
data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'PubChem'].dropna().to_csv('ora_rhythmic_P28 PubChem.csv', index=False)

data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'HMDB'].dropna().to_csv('ora_rhythmic_E19 HMDB.csv', index=False)
data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'HMDB'].dropna().to_csv('ora_rhythmic_P28 HMDB.csv', index=False)


# after review - differential rhythm analysis 
################################################################################################################################
################################################## CIRCA-COMPARE ###############################################################
################################################################################################################################


# CircaCompare AI-modified from original python version, https://github.com/RWParsons/circacompare_py

"""
import numpy as np
from scipy.optimize import least_squares
from scipy import stats
import pandas as pd
from joblib import Parallel, delayed # For parallel processing
import os # To get CPU count

# LOAD my data and change the name of tissue_ages
circacompare_data = pd.read_csv('CircaCompare input SCN raw with NaNs.csv')
tissue_ages = 'SCN P20 vs P28'

# select which groups to compare, here using 'Age' column
group_values = []
for aval, frame in circacompare_data.groupby('Age'):
    group_values.append(aval)

def generate_data(t, k, a, p_cos, noise=0, n_outliers=0, random_state=0): # p_cos is acrophase for cosine
    y = k + a * np.cos(t - p_cos) 
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(t.size)
    outliers = rnd.randint(0, t.size, n_outliers)
    error[outliers] *= 10
    return y + error

# Constants
LOSS = 'linear'
F_SCALE = 1.0
# MAX_ITERATIONS = 500 # This is high for many metabolites. Consider reducing after testing.
# Let's try a more moderate value, user can tune this.
MAX_ITERATIONS = 50 # Max random restarts for optimization. 50 works 

# --- Functions with Analytical Jacobians ---

def fun_circa_single_resid(x, t, y):
    # x[0] = k (mesor), x[1] = a (amplitude), x[2] = p (phase)
    return x[0] + x[1] * np.cos(t - x[2]) - y

def jac_circa_single(x, t, y):
    # x[0]=k, x[1]=a, x[2]=p
    # Residual: r = k + a * cos(t - p) - y
    # dr/dk = 1
    # dr/da = cos(t - p)
    # dr/dp = a * sin(t - p)
    jac = np.empty((t.size, 3))
    cos_t_minus_p = np.cos(t - x[2])
    sin_t_minus_p = np.sin(t - x[2])
    jac[:, 0] = 1.0
    jac[:, 1] = cos_t_minus_p
    jac[:, 2] = x[1] * sin_t_minus_p
    return jac

def fun_circacompare_resid(x, t, y, g):
    # x = [k0, dk, a0, da, p0, dp]
    # K_g = x[0] + x[1]*g
    # A_g = x[2] + x[3]*g
    # P_g = x[4] + x[5]*g
    # Residual: r = K_g + A_g * cos(t - P_g) - y
    return (x[0] + x[1] * g) + (x[2] + x[3] * g) * np.cos(t - (x[4] + x[5] * g)) - y

def jac_circacompare(x, t, y, g):
    # x = [k0, dk, a0, da, p0, dp]
    # K_g = x[0] + x[1]*g
    # A_g = x[2] + x[3]*g
    # P_g = x[4] + x[5]*g
    # Residual: r = K_g + A_g * cos(t - P_g) - y
    # dr/dk0 = 1
    # dr/ddk = g
    # dr/da0 = cos(t - P_g)
    # dr/dda = g * cos(t - P_g)
    # dr/dp0 = A_g * sin(t - P_g)
    # dr/ddp = A_g * g * sin(t - P_g)
    jac = np.empty((t.size, 6))
    A_g = x[2] + x[3] * g
    P_g = x[4] + x[5] * g
    cos_t_minus_Pg = np.cos(t - P_g)
    sin_t_minus_Pg = np.sin(t - P_g)

    jac[:, 0] = 1.0
    jac[:, 1] = g
    jac[:, 2] = cos_t_minus_Pg
    jac[:, 3] = g * cos_t_minus_Pg
    jac[:, 4] = A_g * sin_t_minus_Pg
    jac[:, 5] = A_g * g * sin_t_minus_Pg
    return jac

# MOST ROBUST param_standard_errors with extensive DEBUGGING PRINTS (commented out by default)
def param_standard_errors(optimised_result, y, num_params, met_name_debug="N/A"):
    # TO ENABLE DEBUG PRINTS:
    # 1. Set `debug_metabolite = "YourMetaboliteName"` in the __main__ block.
    # 2. In this function, find the line below:
    #    `# DEBUG_THIS_METABOLITE = (met_name_debug == "YourMetaboliteName_compare" or met_name_debug == "YourMetaboliteName_g0" or met_name_debug == "YourMetaboliteName_g1")`
    #    Uncomment it and replace "YourMetaboliteName" with the actual metabolite name for targeted debugging.
    # 3. Uncomment all the `# if DEBUG_THIS_METABOLITE:` print statements below.

    # Example for enabling debug prints for a metabolite named "Met_0":
    # DEBUG_THIS_METABOLITE = (met_name_debug == "Met_0_compare" or met_name_debug == "Met_0_g0" or met_name_debug == "Met_0_g1" or met_name_debug == "Met_0_analysis" or met_name_debug == "Met_0")
    # For a general debug metabolite name passed:
    # DEBUG_THIS_METABOLITE = (met_name_debug != "N/A" and met_name_debug != "N/A_single" and met_name_debug != "N/A_compare" and met_name_debug != "N/A_analysis" and not met_name_debug.endswith("_g0") and not met_name_debug.endswith("_g1"))

    # Simplified way to enable debug prints if met_name_debug is not one of the generic "N/A..." strings
    # and not just a group suffix. This relies on how met_name_debug is constructed.
    # For the `_process_single_metabolite` function, `met_name_debug` is set to the actual metabolite column name
    # if `debug_single_metabolite_name` matches `met_col`.
    # It is then passed down to circacompare_analysis, circacompare, and circa_single.
    
    # Check if this instance is for the specific metabolite being debugged:
    # This requires `debug_single_metabolite_name` to be set in `__main__` and matched in `_process_single_metabolite`.
    # The `met_name_debug` arg here would then be like "Met_0_compare" or "Met_0_g0".
    # For this example, let's assume we want to debug if the core name matches.
    # debug_core_name = "Met_0" # Change this to your problem metabolite for debugging
    # DEBUG_THIS_METABOLITE = debug_core_name in met_name_debug


    y_flat = np.asarray(y).ravel(); n_obs = y_flat.size
    # if DEBUG_THIS_METABOLITE: print(f"\nDEBUG [{met_name_debug}] --- Entering param_standard_errors ---")
    # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] n_obs={n_obs}, num_params={num_params}")
    if n_obs <= num_params:
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Not enough observations ({n_obs}) for DoF ({num_params}).")
        return np.full(num_params, np.nan)
    ssr = np.nansum(optimised_result.fun ** 2)
    if np.isnan(ssr) or ssr < 0:
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Invalid SSR: {ssr}.")
        return np.full(num_params, np.nan)
    dof = n_obs - num_params
    if dof <= 0:
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Degrees of freedom ({dof}) is not positive.")
        return np.full(num_params, np.nan)
    mse = ssr / dof
    if np.isnan(mse) or mse < 0:
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Invalid MSE calculated: {mse}.")
        return np.full(num_params, np.nan)
    # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] SSR={ssr:.3e}, DoF={dof}, MSE={mse:.3e}")
    jac = optimised_result.jac 
    if jac is None or jac.shape[0] != n_obs or jac.shape[1] != num_params:
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Jacobian from optim_result is invalid or has incorrect shape.")
        return np.full(num_params, np.nan)
    if np.any(~np.isfinite(jac)):
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Jacobian contains NaN/Inf values.")
        return np.full(num_params, np.nan)
    try:
        current_rank = np.linalg.matrix_rank(jac)
        if current_rank < num_params:
            # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Jacobian is rank deficient (rank {current_rank} < {num_params}).")
            return np.full(num_params, np.nan)
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Jacobian rank is {current_rank} (full).")
        jtj = np.dot(jac.T, jac)
        if np.any(~np.isfinite(jtj)):
            # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: J^T J contains NaN/Inf values.")
            return np.full(num_params, np.nan)
        cond_jtj = np.linalg.cond(jtj)
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] J^T J condition number: {cond_jtj:.2e}")
        if cond_jtj > 1e12: 
            # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] High condition number for J^T J. Adding regularization.")
            jtj += np.eye(jtj.shape[0]) * 1e-8 
        inv_jtj = np.linalg.inv(jtj) 
        if np.any(~np.isfinite(inv_jtj)):
            # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Inverse of J^T J contains NaN/Inf values.")
            return np.full(num_params, np.nan)
        diag_inv_jtj = np.diagonal(inv_jtj)
        variances = diag_inv_jtj * mse 
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] MSE={mse:.3e}, diag_inv_jtj={diag_inv_jtj}")
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Calculated parameter variances: {variances}")
        variances[variances < 1e-12] = np.nan 
        param_se = np.sqrt(variances) 
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Calculated SEs: {param_se}. --- Exiting param_standard_errors ---")
    except np.linalg.LinAlgError: 
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: Singular matrix encountered (LinAlgError). --- Exiting param_standard_errors ---")
        param_se = np.full(num_params, np.nan) 
    except Exception as e: 
        # if DEBUG_THIS_METABOLITE: print(f"DEBUG [{met_name_debug}] Exit: An unexpected error occurred: {e}. --- Exiting param_standard_errors ---")
        param_se = np.full(num_params, np.nan) 
    return param_se


def circa_single(t0, y0, loss=LOSS, f_scale=F_SCALE, max_iterations=MAX_ITERATIONS, alpha=0.05):
    result_least_squares = None
    best_ssr = np.inf 

    # Slightly more informed initial guesses
    y0_median = np.median(y0)
    y0_ptp = np.ptp(y0) # peak-to-peak: y0.max() - y0.min()
    if np.isclose(y0_ptp, 0): y0_ptp = np.std(y0) if np.std(y0)>0 else 1.0 # fallback for flat data
    if np.isclose(y0_ptp, 0): y0_ptp = 1.0 # absolute fallback

    for _i in range(max_iterations): 
        k_guess = y0_median + (np.random.rand() - 0.5) * y0_median * 0.2 # Tighter random range around median
        a_guess = np.random.uniform(0.05, 0.75) * y0_ptp # Amplitude as fraction of peak-to-peak
        p_guess = np.random.rand() * 2 * np.pi 
        start_args = np.array([k_guess, max(1e-9, a_guess), p_guess])

        try:
            current_ls_result = least_squares(fun_circa_single_resid, # Use resid version
                                    start_args,
                                    jac=jac_circa_single, # Provide analytical Jacobian
                                    loss=loss,
                                    f_scale=f_scale,
                                    args=(t0, y0),
                                    bounds=([-np.inf, 1e-9, -np.inf], [np.inf, np.inf, np.inf])) 

            if current_ls_result.success:
                current_ssr = np.nansum(current_ls_result.fun ** 2)
                if current_ls_result.x[1] > 1e-9 and current_ssr < best_ssr: 
                    result_least_squares = current_ls_result
                    best_ssr = current_ssr
        except Exception: 
            pass 
    
    # ... (rest of circa_single is the same as before)
    if result_least_squares is None: 
        return {
            'params': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_estimates_raw': [np.nan,np.nan,np.nan], 
            'param_ses': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_cis': {'mesor': [np.nan,np.nan], 'amplitude': [np.nan,np.nan], 'phase': [np.nan,np.nan]},
            'rhythm_f_stat': np.nan, 'rhythm_p_value': np.nan, 'rhythm_detected': False,
            'dof_rhythm_test': [np.nan, np.nan], 'dof_params': np.nan, 'n_obs': y0.size,
            'fit_successful': False, 'message': 'Optimization failed or no valid solution found after max_iterations.'
        }

    estimates = result_least_squares.x
    num_params = len(estimates) 
    dof_params = y0.size - num_params 

    res_se_vals = np.full(num_params, np.nan)
    rhythm_p_value = np.nan
    rhythm_f_stat = np.nan
    dof_rhythm_test = [np.nan, np.nan]
    ssr_full = np.nansum(result_least_squares.fun ** 2)
    mean_y0 = np.mean(y0)
    ssr_reduced = np.nansum((y0 - mean_y0) ** 2) if len(y0)>0 else np.nan

    if dof_params > 0:
        res_se_vals = param_standard_errors(result_least_squares, y0, num_params)
        
        p_full = num_params 
        p_reduced = 1       
        
        df1_rhythm = p_full - p_reduced 
        df2_rhythm = y0.size - p_full   
        dof_rhythm_test = [df1_rhythm, df2_rhythm]

        if df1_rhythm > 0 and df2_rhythm > 0 and not np.isclose(ssr_full, 0, atol=1e-12): 
            mean_sq_effect = (ssr_reduced - ssr_full) / df1_rhythm
            mean_sq_error = ssr_full / df2_rhythm
            if mean_sq_error > 1e-12: 
                 rhythm_f_stat = mean_sq_effect / mean_sq_error
                 if rhythm_f_stat < 0: rhythm_f_stat = 0 
                 rhythm_p_value = stats.f.sf(rhythm_f_stat, df1_rhythm, df2_rhythm)
            else: 
                 rhythm_f_stat = np.inf if mean_sq_effect > 1e-9 else 0 
                 rhythm_p_value = 0.0 if rhythm_f_stat == np.inf else 1.0
        elif np.isclose(ssr_full, 0, atol=1e-12) and (ssr_reduced - ssr_full) > 1e-9 : 
            rhythm_f_stat = np.inf
            rhythm_p_value = 0.0
        else: 
            rhythm_f_stat = np.nan 
            rhythm_p_value = np.nan
    else: 
        rhythm_f_stat = np.nan
        rhythm_p_value = np.nan


    ci_half_width = np.full(num_params, np.nan)
    if dof_params > 0 and not np.all(np.isnan(res_se_vals)):
        valid_se = ~np.isnan(res_se_vals) & (res_se_vals > 1e-9)
        if np.any(valid_se):
            t_crit = stats.t.ppf(1 - alpha / 2, df=dof_params)
            ci_half_width[valid_se] = res_se_vals[valid_se] * t_crit
    
    estimates_norm = estimates.copy()
    estimates_norm[2] = estimates[2] % (2 * np.pi) 

    lower_ci_raw = estimates - ci_half_width
    upper_ci_raw = estimates + ci_half_width
    
    phase_lower_ci_norm = lower_ci_raw[2] % (2 * np.pi)
    phase_upper_ci_norm = upper_ci_raw[2] % (2 * np.pi)

    params_dict = {'mesor': estimates_norm[0], 'amplitude': estimates_norm[1], 'phase': estimates_norm[2]}
    param_ses_dict = {'mesor': res_se_vals[0], 'amplitude': res_se_vals[1], 'phase': res_se_vals[2]}
    param_cis_dict = {
        'mesor': [lower_ci_raw[0], upper_ci_raw[0]],
        'amplitude': [lower_ci_raw[1], upper_ci_raw[1]],
        'phase': [phase_lower_ci_norm, phase_upper_ci_norm] 
    }

    return {
        'params': params_dict,
        'param_estimates_raw': estimates.tolist(), 
        'param_ses': param_ses_dict,
        'param_cis': param_cis_dict,
        'rhythm_f_stat': rhythm_f_stat,
        'rhythm_p_value': rhythm_p_value,
        'rhythm_detected': rhythm_p_value < alpha if not np.isnan(rhythm_p_value) else False,
        'dof_rhythm_test': dof_rhythm_test,
        'dof_params': dof_params, 
        'n_obs': y0.size,
        'fit_successful': True,
        'ssr_full': ssr_full,
        'ssr_reduced': ssr_reduced,
        'optim_status': result_least_squares.status if result_least_squares else -1,
        'message': result_least_squares.message if result_least_squares else 'N/A'
    }


def circacompare(t0, y0, g0, loss=LOSS, f_scale=F_SCALE, max_iterations=MAX_ITERATIONS, alpha=0.05):
    result_least_squares = None
    best_ssr = np.inf

    y0_g0_data = y0[g0 == 0]
    y0_g1_data = y0[g0 == 1]

    if len(y0_g0_data) == 0 or len(y0_g1_data) == 0:
         return {
            'fit_successful': False, 'message': 'One or both groups have no data.',
            'params_group0': None, 'params_group1': None, 'param_differences': None,
            'param_differences_ses': None, 'param_differences_cis': None, 'p_values_difference': None,
            'model_params_coeffs': None, 'model_params_ses': None, 'model_params_cis': None,
            'n_obs_total': y0.size, 'dof_comparison': np.nan
        }

    median_y0_g0 = np.median(y0_g0_data)
    median_y0_g1 = np.median(y0_g1_data)
    ptp_y0_g0 = np.ptp(y0_g0_data) if len(y0_g0_data) > 0 else 1.0
    ptp_y0_g1 = np.ptp(y0_g1_data) if len(y0_g1_data) > 0 else 1.0
    if np.isclose(ptp_y0_g0,0): ptp_y0_g0 = np.std(y0_g0_data) if np.std(y0_g0_data)>0 else 1.0
    if np.isclose(ptp_y0_g1,0): ptp_y0_g1 = np.std(y0_g1_data) if np.std(y0_g1_data)>0 else 1.0
    if np.isclose(ptp_y0_g0,0): ptp_y0_g0 = 1.0 # abs fallback
    if np.isclose(ptp_y0_g1,0): ptp_y0_g1 = 1.0 # abs fallback


    for _i in range(max_iterations):
        k0_guess = median_y0_g0 + (np.random.rand()-0.5) * median_y0_g0 * 0.2
        dk_guess = (median_y0_g1 - median_y0_g0) + (np.random.rand()-0.5) * (median_y0_g1 - median_y0_g0 + 1e-6) * 0.2
        a0_guess = np.random.uniform(0.05, 0.75) * ptp_y0_g0
        da_guess = (np.random.uniform(0.05, 0.75) * ptp_y0_g1) - a0_guess 
        
        start_args = np.array([
            k0_guess, dk_guess,
            max(1e-9, a0_guess), 
            da_guess, 
            np.random.rand() * 2 * np.pi, 
            (np.random.rand() - 0.5) * np.pi 
        ])
        
        if (start_args[2] + start_args[3]) < 1e-9: 
            start_args[3] = -start_args[2] + 1e-9 

        try:
            current_ls_result = least_squares(fun_circacompare_resid, # Use resid version
                                    start_args,
                                    jac=jac_circacompare, # Provide analytical Jacobian
                                    loss=loss,
                                    f_scale=f_scale,
                                    args=(t0, y0, g0),
                                    bounds=([-np.inf,-np.inf, 1e-9, -np.inf, -np.inf, -np.inf], 
                                            [np.inf, np.inf, np.inf,  np.inf,  np.inf,  np.inf])) 
            
            est_x = current_ls_result.x
            if current_ls_result.success and \
               est_x[2] > 1e-9 and \
               (est_x[2] + est_x[3]) > 1e-9: 
                current_ssr = np.nansum(current_ls_result.fun ** 2)
                if current_ssr < best_ssr:
                    result_least_squares = current_ls_result
                    best_ssr = current_ssr
        except Exception:
            pass
    
    # ... (rest of circacompare is the same as before)
    if result_least_squares is None:
        return {
            'fit_successful': False, 'message': 'Optimization failed or no valid solution found after max_iterations.',
            'params_group0': None, 'params_group1': None, 'param_differences': None,
            'param_differences_ses': None, 'param_differences_cis': None, 'p_values_difference': None,
            'model_params_coeffs': None, 'model_params_ses': None, 'model_params_cis': None,
            'n_obs_total': y0.size, 'dof_comparison': np.nan
        }

    estimates = result_least_squares.x 
    num_params = len(estimates) 
    dof_comparison = y0.size - num_params

    res_se_vals = np.full(num_params, np.nan)
    p_values_diff = {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}

    if dof_comparison > 0:
        res_se_vals = param_standard_errors(result_least_squares, y0, num_params)
        
        param_indices_for_diff = {'mesor': 1, 'amplitude': 3, 'phase': 5} 
        for param_name, idx in param_indices_for_diff.items():
            if not np.isnan(res_se_vals[idx]) and res_se_vals[idx] > 1e-9:
                t_stat = estimates[idx] / res_se_vals[idx]
                p_values_diff[param_name] = stats.t.sf(np.abs(t_stat), df=dof_comparison) * 2
            else:
                p_values_diff[param_name] = np.nan
    else: 
        res_se_vals = np.full(num_params, np.nan) 
        p_values_diff = {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}


    ci_half_width = np.full(num_params, np.nan)
    if dof_comparison > 0 and not np.all(np.isnan(res_se_vals)):
        valid_se = ~np.isnan(res_se_vals) & (res_se_vals > 1e-9)
        if np.any(valid_se):
            t_crit = stats.t.ppf(1 - alpha / 2, df=dof_comparison)
            ci_half_width[valid_se] = res_se_vals[valid_se] * t_crit

    raw_lower_ci = estimates - ci_half_width
    raw_upper_ci = estimates + ci_half_width

    k0, dk, a0, da, p0_raw, dp_raw = estimates
    
    mesor_g0 = k0
    amp_g0 = a0
    phase_g0 = p0_raw % (2 * np.pi)

    mesor_g1 = k0 + dk
    amp_g1 = a0 + da 
    phase_g1 = (p0_raw + dp_raw) % (2 * np.pi)

    params_g0_dict = {
        'mesor': mesor_g0, 'amplitude': amp_g0, 'phase': phase_g0,
        'mesor_ci': [raw_lower_ci[0], raw_upper_ci[0]],
        'amplitude_ci': [raw_lower_ci[2], raw_upper_ci[2]],
        'phase_ci': [raw_lower_ci[4] % (2*np.pi), raw_upper_ci[4] % (2*np.pi)] 
    }
    params_g1_dict = { 
        'mesor': mesor_g1, 'amplitude': amp_g1, 'phase': phase_g1,
    }
    
    diff_params_dict = {'mesor': dk, 'amplitude': da, 'phase': dp_raw} 
    diff_ses_dict = {
        'mesor': res_se_vals[1], 'amplitude': res_se_vals[3], 'phase': res_se_vals[5]
    }
    diff_cis_dict = {
        'mesor': [raw_lower_ci[1], raw_upper_ci[1]],
        'amplitude': [raw_lower_ci[3], raw_upper_ci[3]],
        'phase': [raw_lower_ci[5], raw_upper_ci[5]] 
    }

    model_coeffs_dict = {'k0':k0, 'dk':dk, 'a0':a0, 'da':da, 'p0_raw':p0_raw, 'dp_raw':dp_raw}
    model_ses_dict = { name: res_se_vals[i] for i, name in enumerate(['k0', 'dk', 'a0', 'da', 'p0_raw', 'dp_raw']) }
    
    model_cis_list = [[raw_lower_ci[i], raw_upper_ci[i]] for i in range(num_params)]
    model_cis_list[4] = [raw_lower_ci[4] % (2*np.pi), raw_upper_ci[4] % (2*np.pi)] 
    model_cis_dict = { name: model_cis_list[i] for i, name in enumerate(['k0', 'dk', 'a0', 'da', 'p0_raw', 'dp_raw']) }


    results_dict = {
        'params_group0': params_g0_dict,
        'params_group1': params_g1_dict, 
        'param_differences': diff_params_dict,
        'param_differences_ses': diff_ses_dict,
        'param_differences_cis': diff_cis_dict,
        'p_values_difference': p_values_diff,
        
        'model_params_coeffs': model_coeffs_dict,
        'model_params_ses': model_ses_dict,
        'model_params_cis': model_cis_dict,
        
        'n_obs_total': y0.size,
        'dof_comparison': dof_comparison,
        'fit_successful': True,
        'optim_status': result_least_squares.status if result_least_squares else -1,
        'message': result_least_squares.message if result_least_squares else 'N/A'
    }
    return results_dict

def circacompare_analysis(time, measure, group, alpha=0.05):
    analysis_results = {}

    comp_results = circacompare(t0=time, y0=measure, g0=group, alpha=alpha)
    analysis_results['comparison_between_groups'] = comp_results

    t_g0 = time[group == 0]
    y_g0 = measure[group == 0]
    if len(np.unique(t_g0)) > 1 and len(y_g0) > 3 : # Need variation in time and enough points
        single_g0_results = circa_single(t_g0, y_g0, alpha=alpha)
        analysis_results['group0_rhythmicity'] = single_g0_results
    else:
        analysis_results['group0_rhythmicity'] = {
            'fit_successful': False, 
            'message': 'Not enough data points or no time variation for group 0 analysis.',
            'n_obs': len(y_g0),
            # Fill with NaNs for consistency in output structure
            'params': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_ses': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_cis': {'mesor': [np.nan,np.nan], 'amplitude': [np.nan,np.nan], 'phase': [np.nan,np.nan]},
            'rhythm_f_stat': np.nan, 'rhythm_p_value': np.nan, 'rhythm_detected': False,
        }

    t_g1 = time[group == 1]
    y_g1 = measure[group == 1]
    if len(np.unique(t_g1)) > 1 and len(y_g1) > 3:
        single_g1_results = circa_single(t_g1, y_g1, alpha=alpha)
        analysis_results['group1_rhythmicity'] = single_g1_results
    else:
        analysis_results['group1_rhythmicity'] = {
            'fit_successful': False, 
            'message': 'Not enough data points or no time variation for group 1 analysis.',
            'n_obs': len(y_g1),
            'params': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_ses': {'mesor': np.nan, 'amplitude': np.nan, 'phase': np.nan}, 
            'param_cis': {'mesor': [np.nan,np.nan], 'amplitude': [np.nan,np.nan], 'phase': [np.nan,np.nan]},
            'rhythm_f_stat': np.nan, 'rhythm_p_value': np.nan, 'rhythm_detected': False,
        }
        
    return analysis_results

# circacompare_analysis = circacompare_analysis # Function definition is unchanged, just uses the faster internal calls

# --- Helper functions for processing and output formatting (safe_get, get_ci_bounds) ---
def safe_get(data_dict, path, default=np.nan):
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current and current[key] is not None:
            current = current[key]
        else:
            return default
    return current

def get_ci_bounds(data_dict, path_to_ci, default_val=np.nan):
    ci = safe_get(data_dict, path_to_ci, default=[default_val, default_val])
    if isinstance(ci, list) and len(ci) == 2:
        return ci[0], ci[1]
    return default_val, default_val


# --- Wrapper function for parallel processing a single metabolite ---
def _process_single_metabolite(met_col, df_full, time_col, group_col, group0_label, group1_label, period, alpha):
    
    # Processes a single metabolite column from the full DataFrame.
    # This function is designed to be called by joblib.Parallel.

    # print(f"Processing metabolite: {met_col}...") # Printing from parallel jobs can be messy
    df_metabolite = df_full[[time_col, group_col, met_col]].copy()
    df_metabolite.dropna(subset=[met_col], inplace=True)

    if df_metabolite.empty:
        return {'Metabolite': met_col, 'Error': 'No data after NA removal'}

    time_points = df_metabolite[time_col].values
    measurements = df_metabolite[met_col].values
    group_labels = df_metabolite[group_col].values
    time_rad = (time_points / period) * 2 * np.pi

    unique_groups_in_data = np.unique(group_labels)
    expected_groups_present = (group0_label in unique_groups_in_data and group1_label in unique_groups_in_data)
    
    # Handle cases where only one group might be present after filtering for a specific metabolite
    # or if the data for a metabolite only belongs to one group type initially.
    # The circacompare function itself will fail if one group has no data.
    # The circa_single calls can proceed if at least one group has data.

    group_numeric = np.zeros_like(group_labels, dtype=int)
    has_group0 = False
    has_group1 = False
    if group0_label in unique_groups_in_data:
        group_numeric[group_labels == group0_label] = 0
        has_group0 = True
    if group1_label in unique_groups_in_data:
        group_numeric[group_labels == group1_label] = 1 # Will assign 1 if group0_label is not group1_label
        has_group1 = True

    # Critical check for circacompare: needs data from both groups.
    # If not, comparison part of analysis_output will indicate failure.
    # circa_single parts can still run if their respective group has data.
    
    # Ensure all data passed to circacompare_analysis has valid group assignments
    # Filter out rows that don't belong to group0_label or group1_label if other labels exist
    valid_rows = (group_labels == group0_label) | (group_labels == group1_label)
    if not np.all(valid_rows):
        time_rad = time_rad[valid_rows]
        measurements = measurements[valid_rows]
        group_numeric = group_numeric[valid_rows]
        if len(measurements) == 0:
            return {'Metabolite': met_col, 'Error': f'No data after filtering for {group0_label}/{group1_label}'}


    analysis_output = circacompare_analysis(time=time_rad, measure=measurements, group=group_numeric, alpha=alpha)
    
    res_row = {'Metabolite': met_col}
    # ... (the extensive result extraction logic is the same as in your previous process_metabolites_dataframe)
    # Individual rhythmicity - Group 0
    g0_rhythm = analysis_output.get('group0_rhythmicity', {})
    res_row['G0_Fit_Successful'] = safe_get(g0_rhythm, ['fit_successful'], default=False)
    res_row['G0_Message'] = safe_get(g0_rhythm, ['message'], default='N/A')
    res_row['G0_n_obs'] = safe_get(g0_rhythm, ['n_obs'])
    res_row['G0_Rhythm_PValue'] = safe_get(g0_rhythm, ['rhythm_p_value'])
    res_row['G0_Rhythm_FStat'] = safe_get(g0_rhythm, ['rhythm_f_stat'])
    res_row['G0_Rhythm_Detected'] = safe_get(g0_rhythm, ['rhythm_detected'], default=False)
    res_row['G0_Indiv_Mesor'] = safe_get(g0_rhythm, ['params', 'mesor'])
    res_row['G0_Indiv_Amplitude'] = safe_get(g0_rhythm, ['params', 'amplitude'])
    res_row['G0_Indiv_Phase_rad'] = safe_get(g0_rhythm, ['params', 'phase'])
    res_row['G0_Indiv_Mesor_SE'] = safe_get(g0_rhythm, ['param_ses', 'mesor'])
    res_row['G0_Indiv_Amplitude_SE'] = safe_get(g0_rhythm, ['param_ses', 'amplitude'])
    res_row['G0_Indiv_Phase_rad_SE'] = safe_get(g0_rhythm, ['param_ses', 'phase'])
    res_row['G0_Indiv_Mesor_CI_Low'], res_row['G0_Indiv_Mesor_CI_High'] = get_ci_bounds(g0_rhythm, ['param_cis', 'mesor'])
    res_row['G0_Indiv_Amplitude_CI_Low'], res_row['G0_Indiv_Amplitude_CI_High'] = get_ci_bounds(g0_rhythm, ['param_cis', 'amplitude'])
    res_row['G0_Indiv_Phase_rad_CI_Low'], res_row['G0_Indiv_Phase_rad_CI_High'] = get_ci_bounds(g0_rhythm, ['param_cis', 'phase'])

    # Individual rhythmicity - Group 1
    g1_rhythm = analysis_output.get('group1_rhythmicity', {})
    res_row['G1_Fit_Successful'] = safe_get(g1_rhythm, ['fit_successful'], default=False)
    res_row['G1_Message'] = safe_get(g1_rhythm, ['message'], default='N/A')
    res_row['G1_n_obs'] = safe_get(g1_rhythm, ['n_obs'])
    res_row['G1_Rhythm_PValue'] = safe_get(g1_rhythm, ['rhythm_p_value'])
    res_row['G1_Rhythm_FStat'] = safe_get(g1_rhythm, ['rhythm_f_stat'])
    res_row['G1_Rhythm_Detected'] = safe_get(g1_rhythm, ['rhythm_detected'], default=False)
    res_row['G1_Indiv_Mesor'] = safe_get(g1_rhythm, ['params', 'mesor'])
    res_row['G1_Indiv_Amplitude'] = safe_get(g1_rhythm, ['params', 'amplitude'])
    res_row['G1_Indiv_Phase_rad'] = safe_get(g1_rhythm, ['params', 'phase'])
    res_row['G1_Indiv_Mesor_SE'] = safe_get(g1_rhythm, ['param_ses', 'mesor'])
    res_row['G1_Indiv_Amplitude_SE'] = safe_get(g1_rhythm, ['param_ses', 'amplitude'])
    res_row['G1_Indiv_Phase_rad_SE'] = safe_get(g1_rhythm, ['param_ses', 'phase'])
    res_row['G1_Indiv_Mesor_CI_Low'], res_row['G1_Indiv_Mesor_CI_High'] = get_ci_bounds(g1_rhythm, ['param_cis', 'mesor'])
    res_row['G1_Indiv_Amplitude_CI_Low'], res_row['G1_Indiv_Amplitude_CI_High'] = get_ci_bounds(g1_rhythm, ['param_cis', 'amplitude'])
    res_row['G1_Indiv_Phase_rad_CI_Low'], res_row['G1_Indiv_Phase_rad_CI_High'] = get_ci_bounds(g1_rhythm, ['param_cis', 'phase'])

    # Comparison results
    comp = analysis_output.get('comparison_between_groups', {})
    res_row['Compare_Fit_Successful'] = safe_get(comp, ['fit_successful'], default=False)
    res_row['Compare_Message'] = safe_get(comp, ['message'], default='N/A')
    res_row['Compare_Total_n_obs'] = safe_get(comp, ['n_obs_total'])
    res_row['Comp_G0_Mesor'] = safe_get(comp, ['params_group0', 'mesor'])
    res_row['Comp_G0_Amplitude'] = safe_get(comp, ['params_group0', 'amplitude'])
    res_row['Comp_G0_Phase_rad'] = safe_get(comp, ['params_group0', 'phase'])
    res_row['Comp_G0_Mesor_CI_Low'], res_row['Comp_G0_Mesor_CI_High'] = get_ci_bounds(comp, ['params_group0', 'mesor_ci'])
    res_row['Comp_G0_Amplitude_CI_Low'], res_row['Comp_G0_Amplitude_CI_High'] = get_ci_bounds(comp, ['params_group0', 'amplitude_ci'])
    res_row['Comp_G0_Phase_rad_CI_Low'], res_row['Comp_G0_Phase_rad_CI_High'] = get_ci_bounds(comp, ['params_group0', 'phase_ci'])
    res_row['Comp_G1_Mesor'] = safe_get(comp, ['params_group1', 'mesor'])
    res_row['Comp_G1_Amplitude'] = safe_get(comp, ['params_group1', 'amplitude'])
    res_row['Comp_G1_Phase_rad'] = safe_get(comp, ['params_group1', 'phase'])
    res_row['Diff_Mesor_Est'] = safe_get(comp, ['param_differences', 'mesor'])
    res_row['Diff_Mesor_SE'] = safe_get(comp, ['param_differences_ses', 'mesor'])
    res_row['Diff_Mesor_PValue'] = safe_get(comp, ['p_values_difference', 'mesor'])
    res_row['Diff_Mesor_CI_Low'], res_row['Diff_Mesor_CI_High'] = get_ci_bounds(comp, ['param_differences_cis', 'mesor'])
    res_row['Diff_Amplitude_Est'] = safe_get(comp, ['param_differences', 'amplitude'])
    res_row['Diff_Amplitude_SE'] = safe_get(comp, ['param_differences_ses', 'amplitude'])
    res_row['Diff_Amplitude_PValue'] = safe_get(comp, ['p_values_difference', 'amplitude'])
    res_row['Diff_Amplitude_CI_Low'], res_row['Diff_Amplitude_CI_High'] = get_ci_bounds(comp, ['param_differences_cis', 'amplitude'])
    res_row['Diff_Phase_rad_Est'] = safe_get(comp, ['param_differences', 'phase']) 
    res_row['Diff_Phase_rad_SE'] = safe_get(comp, ['param_differences_ses', 'phase'])
    res_row['Diff_Phase_rad_PValue'] = safe_get(comp, ['p_values_difference', 'phase'])
    res_row['Diff_Phase_rad_CI_Low'], res_row['Diff_Phase_rad_CI_High'] = get_ci_bounds(comp, ['param_differences_cis', 'phase'])

    return res_row


def process_metabolites_dataframe_parallel(df, time_col='Time', group_col='Age', 
                                           group0_label=group_values[0], group1_label=group_values[1], 
                                           period=24, alpha=0.05, n_jobs=-1):
    metabolite_cols = [col for col in df.columns if col not in [time_col, group_col]]
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() # Use all available CPUs
        print(f"Using {n_jobs} cores for parallel processing.")
    elif n_jobs == 1:
        print("Using 1 core (serial processing).")
    else:
        print(f"Using {n_jobs} cores for parallel processing.")


    # Use joblib to parallelize the loop
    # Setting backend to 'loky' (default) is generally robust.
    # 'multiprocessing' can sometimes have issues with complex objects or in certain environments (e.g. interactive)
    # For heavy numerical tasks like this, 'loky' or 'multiprocessing' are usually good.
    # Using 'threading' backend is not suitable for CPU-bound tasks like this due to Python's GIL.
    results_list = Parallel(n_jobs=n_jobs, verbose=10)( # verbose=10 shows progress
        delayed(_process_single_metabolite)(
            met_col, df, time_col, group_col, group0_label, group1_label, period, alpha
        ) for met_col in metabolite_cols
    )
    
    return pd.DataFrame(results_list)


if __name__ == '__main__':
    # --- Example Usage with a Dummy DataFrame ---
    n_timepoints_unique = 7 
    timepoints_hrs = [0, 4, 8, 12, 16, 20, 24] 
    n_replicates = 5
    ages = ['E19', 'P28']
    
    data_list = []
    np.random.seed(42) 
    
    # Generate data for a few metabolites to test
    # To test with 800, uncomment the larger n_metabolites_dummy
    # n_metabolites_dummy = 800 
    n_metabolites_dummy = 5 # Small number for quick test
    dummy_metabolite_names = [f"Met_{i}" for i in range(n_metabolites_dummy)]


    for age in ages:
        for t_hr in timepoints_hrs:
            for _ in range(n_replicates):
                row_data = {'Age': age, 'Time': t_hr}
                t_rad_for_gen = (t_hr / 24.0) * 2 * np.pi 
                
                for met_idx, met_name in enumerate(dummy_metabolite_names):
                    # Vary parameters slightly for each dummy metabolite
                    k_base = 1500000 + met_idx * 10000
                    a_base = 500000 - met_idx * 5000
                    p_base = np.pi + met_idx * 0.1

                    k_met = k_base if age == 'E19' else k_base * 1.2
                    a_met = a_base if age == 'E19' else a_base * 0.7
                    p_met = p_base if age == 'E19' else (p_base + np.pi/2) % (2*np.pi)
                    
                    met_val = generate_data(t=np.array([t_rad_for_gen]), k=k_met, a=a_met, p_cos=p_met, noise=k_base*0.2)[0]
                    row_data[met_name] = met_val
                data_list.append(row_data)

    dummy_df = pd.DataFrame(data_list)
    print("Dummy DataFrame Head:")
    print(dummy_df.head())
    print(f"\nTotal rows in dummy_df: {len(dummy_df)}, Total metabolites: {n_metabolites_dummy}")

    import time
    start_time = time.time()

    # Process the dummy DataFrame in parallel
    results_df = process_metabolites_dataframe_parallel(
                                               dummy_df, 
                                               time_col='Time', 
                                               group_col='Age', 
                                               group0_label=group_values[0], 
                                               group1_label=group_values[1], 
                                               period=24, 
                                               alpha=0.05,
                                               n_jobs=-1) # Use -1 for all cores, or specify a number like 4

    end_time = time.time()
    print(f"\nProcessing {n_metabolites_dummy} metabolites took {end_time - start_time:.2f} seconds.")

    print("\n--- CircaCompare Results for Metabolites (Parallel) ---")
    pd.set_option('display.max_columns', None) 
    pd.set_option('display.width', 1000)
    print(results_df.head())

    # results_df.to_csv("circacompare_metabolite_results_parallel.csv", index=False)
    # print("\nResults saved to circacompare_metabolite_results_parallel.csv")



start_time = time.time()

# Process the dummy DataFrame in parallel
results_df = process_metabolites_dataframe_parallel(
                                           circacompare_data, 
                                           time_col='Time', 
                                           group_col='Age', 
                                           group0_label=group_values[0], 
                                           group1_label=group_values[1], 
                                           period=24, 
                                           alpha=0.05,
                                           n_jobs=-1) # Use -1 for all cores, or specify a number like 4

end_time = time.time()
results_df.to_csv(f'CircaCompare_{tissue_ages}_results_df.csv')
print(f"\nProcessing {n_metabolites_dummy} metabolites took {end_time - start_time:.2f} seconds.")

# Apply FDR correction to all P values, can modify alpha, method='fdr_bh','fdr_by', 'holm', 'bonferroni'
# Example: FDR for Mesor Difference P-values
# 1. Get the raw p-values for mesor difference, dropping NaNs as multipletests requires finite values
raw_pvals_mesor_diff = results_df['Diff_Mesor_PValue'].dropna()

if not raw_pvals_mesor_diff.empty:
    reject_mesor, pvals_corrected_mesor, _, _ = multipletests(
        raw_pvals_mesor_diff, 
        alpha=0.05,       # Your desired overall FDR level
        method='fdr_bh'   # Benjamini-Hochberg
    )
    
    # Add the corrected p-values back to the DataFrame
    # Create a new column initialized with NaN
    results_df['Diff_Mesor_PValue_BH_FDR'] = np.nan
    # Place the corrected p-values back matching the original non-NaN indices
    results_df.loc[raw_pvals_mesor_diff.index, 'Diff_Mesor_PValue_BH_FDR'] = pvals_corrected_mesor
    
    print("\nCorrected P-values for Mesor Difference (first 10 shown):")
    print(results_df[['Metabolite', 'Diff_Mesor_PValue', 'Diff_Mesor_PValue_BH_FDR']].head(10))
    print(f"Number of mesor differences significant after BH-FDR (at alpha=0.05): {np.sum(reject_mesor)}")
else:
    print("No valid raw p-values found for Mesor Difference to correct.")

# Example: FDR for Amplitude Difference P-values
raw_pvals_amp_diff = results_df['Diff_Amplitude_PValue'].dropna()
if not raw_pvals_amp_diff.empty:
    reject_amp, pvals_corrected_amp, _, _ = multipletests(raw_pvals_amp_diff, alpha=0.05, method='fdr_bh')
    results_df['Diff_Amplitude_PValue_BH_FDR'] = np.nan
    results_df.loc[raw_pvals_amp_diff.index, 'Diff_Amplitude_PValue_BH_FDR'] = pvals_corrected_amp
    print("\nCorrected P-values for Amplitude Difference (first 10 shown):")
    print(results_df[['Metabolite', 'Diff_Amplitude_PValue', 'Diff_Amplitude_PValue_BH_FDR']].head(10))

# Example: FDR for Phase Difference P-values
raw_pvals_phase_diff = results_df['Diff_Phase_rad_PValue'].dropna()
if not raw_pvals_phase_diff.empty:
    reject_phase, pvals_corrected_phase, _, _ = multipletests(raw_pvals_phase_diff, alpha=0.05, method='fdr_bh')
    results_df['Diff_Phase_rad_PValue_BH_FDR'] = np.nan
    results_df.loc[raw_pvals_phase_diff.index, 'Diff_Phase_rad_PValue_BH_FDR'] = pvals_corrected_phase
    print("\nCorrected P-values for Phase Difference (first 10 shown):")
    print(results_df[['Metabolite', 'Diff_Phase_rad_PValue', 'Diff_Phase_rad_PValue_BH_FDR']].head(10))

# Example: FDR for Group 0 Rhythmicity P-values
raw_pvals_g0_rhythm = results_df['G0_Rhythm_PValue'].dropna()
if not raw_pvals_g0_rhythm.empty:
    reject_g0_rhythm, pvals_corrected_g0_rhythm, _, _ = multipletests(raw_pvals_g0_rhythm, alpha=0.05, method='fdr_bh')
    results_df['G0_Rhythm_PValue_BH_FDR'] = np.nan
    results_df.loc[raw_pvals_g0_rhythm.index, 'G0_Rhythm_PValue_BH_FDR'] = pvals_corrected_g0_rhythm
    print("\nCorrected P-values for Group 0 Rhythmicity (first 10 shown):")
    print(results_df[['Metabolite', 'G0_Rhythm_PValue', 'G0_Rhythm_PValue_BH_FDR']].head(10))

# Example: FDR for Group 1 Rhythmicity P-values
raw_pvals_g1_rhythm = results_df['G1_Rhythm_PValue'].dropna()
if not raw_pvals_g1_rhythm.empty:
    reject_g1_rhythm, pvals_corrected_g1_rhythm, _, _ = multipletests(raw_pvals_g1_rhythm, alpha=0.05, method='fdr_bh')
    results_df['G1_Rhythm_PValue_BH_FDR'] = np.nan
    results_df.loc[raw_pvals_g1_rhythm.index, 'G1_Rhythm_PValue_BH_FDR'] = pvals_corrected_g1_rhythm
    print("\nCorrected P-values for Group 1 Rhythmicity (first 10 shown):")
    print(results_df[['Metabolite', 'G1_Rhythm_PValue', 'G1_Rhythm_PValue_BH_FDR']].head(10))


# Save the DataFrame with corrected p-values
results_df.to_csv(f'CircaCompare_{tissue_ages}_results_with_FDR.csv', index=False)
print("\nResults with FDR corrected p-values saved.")

# check how many are rhythmic in each group
# results_df.loc[results_df['G0_Rhythm_Detected'] == True, 'Metabolite']
# results_df.loc[results_df['G1_Rhythm_Detected'] == True, 'Metabolite']                                              

"""
# Reload data
data = pd.read_csv('_data_all_with_calculations.csv')

# CORRECT Lysine is not Peptide but Amino acid
data.loc[data.Name == 'Lysine', 'Class'] = 'Amino acid'


# LOAD CircaCompare SCN results
results_df = pd.read_csv('CircaCompare_SCN E19 vs P28_results_with_FDR.csv')
# Merge 
data[['CircaComp_Pval_E19', 'CircaComp_Pval_P28', 'CircaComp_Mesor_Pval_E19xP28', 'CircaComp_Amp_Pval_E19xP28', 'CircaComp_Phase_Pval_E19xP28']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_SCN E19 vs P02_results_with_FDR.csv')
# Merge 
data[['CircaComp_Pval_E19', 'CircaComp_Pval_P02', 'CircaComp_Mesor_Pval_E19xP02', 'CircaComp_Amp_Pval_E19xP02', 'CircaComp_Phase_Pval_E19xP02']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_SCN P02 vs P10_results_with_FDR.csv')
# Merge 
data[['CircaComp_Pval_P02', 'CircaComp_Pval_P10', 'CircaComp_Mesor_Pval_P02xP10', 'CircaComp_Amp_Pval_P02xP10', 'CircaComp_Phase_Pval_P02xP10']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_SCN P10 vs P20_results_with_FDR.csv')
# Merge 
data[['CircaComp_Pval_P10', 'CircaComp_Pval_P20', 'CircaComp_Mesor_Pval_P10xP20', 'CircaComp_Amp_Pval_P10xP20', 'CircaComp_Phase_Pval_P10xP20']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_SCN P20 vs P28_results_with_FDR.csv')
# Merge 
data[['CircaComp_Pval_P20', 'CircaComp_Pval_P28', 'CircaComp_Mesor_Pval_P20xP28', 'CircaComp_Amp_Pval_P20xP28', 'CircaComp_Phase_Pval_P20xP28']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]


# LOAD CircaCompare PLS results
results_df = pd.read_csv('CircaCompare_PLS E19 vs P28_results_with_FDR.csv')
# Merge 
data[['PLS_CircaComp_Pval_E19', 'PLS_CircaComp_Pval_P28', 'PLS_CircaComp_Mesor_Pval_E19xP28', 'PLS_CircaComp_Amp_Pval_E19xP28', 'PLS_CircaComp_Phase_Pval_E19xP28']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_PLS E19 vs P02_results_with_FDR.csv')
# Merge 
data[['PLS_CircaComp_Pval_E19', 'PLS_CircaComp_Pval_P02', 'PLS_CircaComp_Mesor_Pval_E19xP02', 'PLS_CircaComp_Amp_Pval_E19xP02', 'PLS_CircaComp_Phase_Pval_E19xP02']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_PLS P02 vs P10_results_with_FDR.csv')
# Merge 
data[['PLS_CircaComp_Pval_P02', 'PLS_CircaComp_Pval_P10', 'PLS_CircaComp_Mesor_Pval_P02xP10', 'PLS_CircaComp_Amp_Pval_P02xP10', 'PLS_CircaComp_Phase_Pval_P02xP10']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_PLS P10 vs P20_results_with_FDR.csv')
# Merge 
data[['PLS_CircaComp_Pval_P10', 'PLS_CircaComp_Pval_P20', 'PLS_CircaComp_Mesor_Pval_P10xP20', 'PLS_CircaComp_Amp_Pval_P10xP20', 'PLS_CircaComp_Phase_Pval_P10xP20']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]

results_df = pd.read_csv('CircaCompare_PLS P20 vs P28_results_with_FDR.csv')
# Merge 
data[['PLS_CircaComp_Pval_P20', 'PLS_CircaComp_Pval_P28', 'PLS_CircaComp_Mesor_Pval_P20xP28', 'PLS_CircaComp_Amp_Pval_P20xP28', 'PLS_CircaComp_Phase_Pval_P20xP28']] = results_df.iloc[:, :][['G0_Rhythm_PValue_BH_FDR', 'G1_Rhythm_PValue_BH_FDR', 'Diff_Mesor_PValue_BH_FDR', 'Diff_Amplitude_PValue_BH_FDR', 'Diff_Phase_rad_PValue_BH_FDR']]


# create column with True or False for 
data['E19_P28_cc_parameters_different'] = ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq) | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))
data['PLS_E19_P28_cc_parameters_different'] = ((data['PLS_CircaComp_Mesor_Pval_E19xP28'] < bestq) | (data['PLS_CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['PLS_CircaComp_Phase_Pval_E19xP28'] < bestq))


# CHECK how CircaCompare and eJTK matches
E19_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'Name']
#E19_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq), 'Name']
P28_set = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Name']
#P28_set = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq), 'Name']
E19P28_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05) & (data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Name']
E19nP28_set = data.iloc[:-5, :].loc[((data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05)) & ((data['P28_emp p BH Corrected'] >= bestq) | (data['1w_ANOVA_P28'] >= 0.05)), 'Name']
P28nE19_set = data.iloc[:-5, :].loc[((data['E19_emp p BH Corrected'] >= bestq) | (data['1w_ANOVA_E19'] >= 0.05)) & ((data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05)), 'Name']

E19_set_cc = data.iloc[:-5, :].loc[(data['CircaComp_Pval_E19'] < bestq) , 'Name']
P02_set_cc = data.iloc[:-5, :].loc[(data['CircaComp_Pval_P02'] < bestq) , 'Name']
P10_set_cc = data.iloc[:-5, :].loc[(data['CircaComp_Pval_P10'] < bestq) , 'Name']
P20_set_cc = data.iloc[:-5, :].loc[(data['CircaComp_Pval_P20'] < bestq) , 'Name']
P28_set_cc = data.iloc[:-5, :].loc[(data['CircaComp_Pval_P28'] < bestq) , 'Name']

E19_E19cc_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05) & (data['CircaComp_Pval_E19'] < bestq) & (data['1w_ANOVA_E19'] < 0.05), 'Name']
P28_P28cc_set = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05) & (data['CircaComp_Pval_P28'] < bestq) & (data['1w_ANOVA_P28'] < 0.05), 'Name']
Venn(len(E19_set), len(E19_set_cc), len(E19_E19cc_set), labels=('E19 bestq', 'E19 circaCompare with ANOVA'), mydir=mydir)
Venn(len(P28_set), len(P28_set_cc), len(P28_P28cc_set), labels=('P28 bestq', 'P28 circaCompare with ANOVA'), mydir=mydir)



# CC parameters included E19 vs P28
E19cc_set_rhythm_and_param = data.iloc[:-5, :].loc[(data['CircaComp_Pval_E19'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq)), 'Name']

# slightly different when compareing parameters between E19 and P02
E19cc_set_rhythm_and_param2 = data.iloc[:-5, :].loc[(data['CircaComp_Pval_E19'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP02'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP02'] < bestq) | (data['CircaComp_Phase_Pval_E19xP02'] < bestq)), 'Name']

# CC parameters included P28 vs E19
P28cc_set_rhythm_and_param = data.iloc[:-5, :].loc[(data['CircaComp_Pval_P28'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq)), 'Name']


E19_E19cc_set = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05) & ((data['CircaComp_Pval_E19'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name']

Venn(len(E19_set), len(E19cc_set_rhythm_and_param), len(E19_E19cc_set), labels=('E19 bestq', 'E19 circaCompare with parameters'), mydir=mydir)

P28_P28cc_set = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05) & ((data['CircaComp_Pval_P28'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name']

Venn(len(P28_set), len(P28cc_set_rhythm_and_param), len(P28_P28cc_set), labels=('P28 bestq', 'P28 circaCompare with parameters'), mydir=mydir)


value_count_plots_filter_sortcounts(data.iloc[:-5, :], 'Class', pval_f1=['E19_emp p BH Corrected', 'P28_emp p BH Corrected'], mydir=mydir, pval_f2=['1w_ANOVA_E19', '1w_ANOVA_P28'],  alpha1=bestq, alpha2=bestq)

value_count_plots_filter_sortcounts(data.iloc[:-5, :], 'Class', pval_f1=['CircaComp_Pval_E19', 'CircaComp_Pval_P28'], mydir=mydir, pval_f2=['1w_ANOVA_E19', '1w_ANOVA_P28'],  alpha1=bestq, alpha2=bestq)


# Value counts plot for 4 pvalues from CircaCompare
def value_count_plots_filter_sortcounts_mutlipval(data, col, pval_f1, mydir, pval_f2=np.nan, pval_f3=np.nan, pval_f4=np.nan, alpha=0.05, how_many=3):
    sns.set_context("paper", font_scale=0.9) 
   
    df1 = data.loc[(data[pval_f1[0]] < alpha) & ((data[pval_f2[0]] < alpha) & ((data[pval_f3[0]] < alpha) | (data[pval_f4[0]] < alpha))), col]
    df3 = data.loc[(data[pval_f1[1]] < alpha) & ((data[pval_f2[1]] < alpha) & ((data[pval_f3[1]] < alpha) | (data[pval_f4[1]] < alpha))), col]
    
    
    df2 = data.loc[(data[pval_f1[0]] < alpha) & ((data[pval_f2[0]] < alpha) & ((data[pval_f3[0]] < alpha) | (data[pval_f4[0]] < alpha)))
                   & (data[pval_f1[1]] < alpha) & ((data[pval_f2[1]] < alpha) & ((data[pval_f3[1]] < alpha) | (data[pval_f4[1]] < alpha))), col]                       
    
    fig, ax = plt.subplots(1, 3, sharey=True)  # figsize=(8, 8)
    g1 = sns.barplot(x='index', y=col, data=df1.value_counts(ascending=True)[lambda x: x >= how_many].to_frame().reset_index(), ax=ax[0])
    g1.set(xlabel=None)
    g1.set(ylabel=None)
    g1.set_xticklabels(g1.get_xticklabels(),rotation=90)
    g1.set(title=f'{pval_f1[0][0]}_group')
    try:          
        g2 = sns.barplot(x='index', y=col, data=df2.value_counts(ascending=True)[lambda x: x >= how_many].to_frame().reset_index(), ax=ax[1])
        g2.set(xlabel=None)
        g2.set(ylabel=None)
        g2.set_xticklabels(g2.get_xticklabels(),rotation=90)
        g2.set(title=f'{pval_f1[0][0]}{pval_f1[1][0]}_group')
    except ValueError:
        pass        
    g3 = sns.barplot(x='index', y=col, data=df3.value_counts(ascending=True)[lambda x: x >= how_many].to_frame().reset_index(), ax=ax[2])
    g3.set(xlabel=None)
    g3.set(ylabel=None)
    g3.set_xticklabels(g3.get_xticklabels(),rotation=90)
    g3.set(title=f'{pval_f1[1][0]}_group')

    plt.rcParams['svg.fonttype'] = 'none'
    plt.savefig(f'{mydir}\\_{pval_f1[0]}_group_mutipval.svg', format = 'svg', bbox_inches = 'tight')
    plt.savefig(f'{mydir}\\_{pval_f1[0]}_group_mutipval.png', format = 'png', bbox_inches = 'tight') 
    plt.show()
    plt.clf()
    plt.close()  



# CC parameters included E19 vs P28, Fig. S5
# E19
E19ccparams_eJTK = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq)), 'Name']

E19_E19ccparams_eJTK = data.iloc[:-5, :].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < 0.05) & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name']

Venn(len(E19_set), len(E19ccparams_eJTK), len(E19_E19ccparams_eJTK), labels=('E19 bestq', 'E19 eJTK with CCparameters'), mydir=mydir)

# P28
P28ccparams_eJTK = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq)), 'Name']

P28_P28ccparams_eJTK = data.iloc[:-5, :].loc[(data['P28_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_P28'] < 0.05) & ((data['P28_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name']

Venn(len(P28_set), len(P28ccparams_eJTK), len(P28_P28ccparams_eJTK), labels=('P28 bestq', 'P28 eJTK with CCparameters'), mydir=mydir)
# Fig. S5
value_count_plots_filter_sortcounts_mutlipval(data.iloc[:-5, :], 'Class', pval_f1=['E19_emp p BH Corrected', 'P28_emp p BH Corrected'], mydir=mydir, pval_f2=['CircaComp_Mesor_Pval_E19xP28', 'CircaComp_Mesor_Pval_E19xP28'], 
                                    pval_f3=['CircaComp_Amp_Pval_E19xP28', 'CircaComp_Amp_Pval_E19xP28'], pval_f4=['CircaComp_Phase_Pval_E19xP28', 'CircaComp_Phase_Pval_E19xP28'],  alpha=bestq)



# uses CircaCompare for E19, P28 SCN and PLS
def plot_gene_group_select_tissue_cc(data, name_list, title, tissue='SCN', annotation=True, describe=False, des_list=[], y_norm=True, errorbar='sem', fnt=80, ano2w=True, style='plain', pad=-2):  # errorbar='std' style='sci'
    
    ### To save as vector svg with fonts editable in Corel ###
    # mpl.use('svg')                                                                          #import matplotlib as mpl
    # new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
    # mpl.rcParams.update(new_rc_params)

    counter = 0
    Nr = len(name_list)    
    Nc = math.sqrt(Nr) #find the square root of x
    Nw = int(Nr/Nc)
    if Nc.is_integer() == False: #check y is not a whole number
        Nw = math.ceil(Nc)
        Nc = int(round(Nc, 0))        
    Nc = int(Nc)
    figsize = (Nc*3, Nw*3)
    
    fig, axs = plt.subplots(Nw, Nc, sharex=True, figsize=figsize)
    for i in range(Nw):
        for j in range(Nc):                
            
            if Nw*Nc > Nr:
                if counter == Nr:
                    break  
                            
            df=pd.DataFrame()
            # df['Name'] = data['Name']            
            # df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']  
            if tissue == 'SCN':
                df = data.loc[:, 'log2_E19_ZT00_1':'log2_P28_ZT24_5']
            if tissue == 'PLS':
                df = data.loc[:, 'log2_PLS_E19_ZT00_1':'log2_PLS_P28_ZT24_5']      
            if tissue == 'LIV':
                df = data.loc[:, 'log2_LIV_E19_ZT00_1':'log2_LIV_P28_ZT24_5']  

            df.insert(0, 'Name', data['Name'])             
            name = name_list[counter]
            # Select columns with the specified data label
            dfn = df.loc[df['Name'] == name, df.columns[1:]]               
            
            # Reshape the data into long format using melt
            data_melted = dfn.melt(var_name="Timepoint", value_name="Value")    
            # Split the Timepoint column into two separate columns for time and replicate 
            # data_melted[["Time", "Replicate"]] = data_melted["Timepoint"].str.split(".", expand=True)
            data_melted['Group'] = data_melted['Timepoint'].str.extract(r'_(E\d+|P\d+)_')
            data_melted[["Time", "Replicate"]] = data_melted['Timepoint'].str.extract(r'_ZT(\d+)_([\d]+)')  
            # Convert Time and Replicate columns to numeric types
            data_melted["Time"] = pd.to_numeric(data_melted["Time"])
            data_melted["Replicate"] = pd.to_numeric(data_melted["Replicate"])      

            # pvalues
            if tissue == 'SCN':
                pC = str(round(float(data.loc[data.Name == name, 'E19_emp p BH Corrected']), 4)) # uses eJTK fro SCN
                pX = str(round(float(data.loc[data.Name == name, 'P28_emp p BH Corrected']), 4))  
                p02 = str(round(float(data.loc[data.Name == name, 'P02_emp p BH Corrected']), 4))
                p10 = str(round(float(data.loc[data.Name == name, 'P10_emp p BH Corrected']), 4))
                p20 = str(round(float(data.loc[data.Name == name, 'P20_emp p BH Corrected']), 4))
                
                # True or False - is any CircaCompare parameters different between E19 and P28?
                # BE CAREFUL THIS IS BUGGY - need to use str, array, ==
                params_significantly_differ = str(data.loc[data.Name == name, 'E19_P28_cc_parameters_different'].array[0])                
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))

            if tissue == 'PLS':
                pC = str(round(float(data.loc[data.Name == name, 'E19_PLS_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_PLS_emp p BH Corrected']), 4))  
                p02 = str(round(float(data.loc[data.Name == name, 'PLS_CircaComp_Pval_P02']), 4)) # for PLS P02-20, instead of eJTK, uses CircaCompare for rhythmicity test
                p10 = str(round(float(data.loc[data.Name == name, 'PLS_CircaComp_Pval_P10']), 4))
                p20 = str(round(float(data.loc[data.Name == name, 'PLS_CircaComp_Pval_P20']), 4))
                
                # True or False - is any CircaCompare parameters different between PLS E19 and P28?
                PLS_params_significantly_differ = str(data.loc[data.Name == name, 'PLS_E19_P28_cc_parameters_different'].array[0])
                
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))
                
            if tissue == 'LIV':
                pC = str(round(float(data.loc[data.Name == name, 'E19_LIV_emp p BH Corrected']), 4))
                pX = str(round(float(data.loc[data.Name == name, 'P28_LIV_emp p BH Corrected']), 4))  
                
                params_significantly_differ = 'True'
    
                if annotation == True:
                    axs[i, j].annotate(f'E19 p = {pC}\n P28 p = {pX}\n', xy=(1, 1), xycoords='axes fraction', 
                    xytext=(-5, -100), textcoords='offset points', horizontalalignment='right', verticalalignment='top', fontsize=round(int(fnt/Nc)))


            P28 = data_melted.loc[data_melted['Group'] == 'P28'][['Time', 'Value']]
            # Group the data by 'time' and calculate the mean and standard deviation
            grouped_P28 = P28.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            # Extract the values for plotting
            time_points = grouped_P28.index
            mean_values_P28 = grouped_P28['Value', 'mean']
            std_values_P28 = grouped_P28['Value', f'{errorbar}']            
            # Plot the line plot with error bars
            if tissue == 'SCN':
                if float(pX) < 0.05 and params_significantly_differ  == 'True':
                    line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4) # full line

                else:
                    line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, ls='--', capsize=4)                    
                    
            if tissue == 'PLS':                    
                if float(pX) < 0.05 and PLS_params_significantly_differ  == 'True':
                    line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, capsize=4) # full line
                else:
                    line1 = axs[i, j].errorbar(time_points, mean_values_P28, yerr=std_values_P28, color='tomato', linewidth=2, ls='--', capsize=4)                 

            E19 = data_melted.loc[data_melted['Group'] == 'E19'][['Time', 'Value']]
            grouped_E19 = E19.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_E19 = grouped_E19['Value', 'mean']
            std_values_E19 = grouped_E19['Value', f'{errorbar}']
            if tissue == 'SCN':
                if float(pC) < 0.05 and params_significantly_differ  == 'True':
                    line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, capsize=4) # full line
                else:
                    line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_E19, color='slateblue', linewidth=2, ls='--', capsize=4)
            
            if tissue == 'PLS':
                if float(pC) < 0.05 and PLS_params_significantly_differ  == 'True':
                    line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_P28, color='slateblue', linewidth=2, capsize=4) # full line
                else:
                    line2 = axs[i, j].errorbar(time_points, mean_values_E19, yerr=std_values_P28, color='slateblue', linewidth=2, ls='--', capsize=4)   
            
            P02 = data_melted.loc[data_melted['Group'] == 'P02'][['Time', 'Value']]
            grouped_P02 = P02.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P02 = grouped_P02['Value', 'mean']
            std_values_P02 = grouped_P02['Value', f'{errorbar}']      
            if float(p02) < 0.05:
                line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, capsize=4) # full line
            else:
                line3 = axs[i, j].errorbar(time_points, mean_values_P02, yerr=std_values_P02, color='grey', linewidth=2, ls='--', capsize=4)  

            P10 = data_melted.loc[data_melted['Group'] == 'P10'][['Time', 'Value']]
            grouped_P10 = P10.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P10 = grouped_P10['Value', 'mean']
            std_values_P10 = grouped_P10['Value', f'{errorbar}']      
            if float(p10) < 0.05:
                line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, capsize=4)  # full line
            else:
                line4 = axs[i, j].errorbar(time_points, mean_values_P10, yerr=std_values_P10, color='green', linewidth=2, ls='--', capsize=4)   

            P20 = data_melted.loc[data_melted['Group'] == 'P20'][['Time', 'Value']]
            grouped_P20 = P20.groupby('Time').agg({'Value': ['mean', f'{errorbar}']})
            mean_values_P20 = grouped_P20['Value', 'mean']
            std_values_P20 = grouped_P20['Value', f'{errorbar}']      
            if float(p20) < 0.05:
                line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, capsize=4)   # full line
            else:
                line5 = axs[i, j].errorbar(time_points, mean_values_P20, yerr=std_values_P20, color='orange', linewidth=2, ls='--', capsize=4)   
            
            if y_norm == True:
                ymin = math.floor(data_melted['Value'].min())
                axs[i, j].set_ylim(ymin)
               
            # if ano2w == True:
            #     model = ols('Value ~ Time + Group + Time:Group', data=data_melted).fit()
            #     result = sm.stats.anova_lm(model, type=2)
            #     p2wano = result['PR(>F)']['Group']                
    
            # # Add a title and axis labels            
            axs[i, j].ticklabel_format(axis='y', style=style, scilimits=(0,0), useMathText=True, useOffset=True) # force scientific notation of all y labels      
            axs[i, j].tick_params(axis='y', which='major', pad=pad, width=1) # move lables closer to y ticks                        
            axs[i, j].set_xticks([0, 4, 8, 12, 16, 20, 24])
            axs[i, j].tick_params(axis='x', which='major', pad=0, width=1)
            # axs[i, j].set_yticks([])
            axs[i, j].spines['top'].set_visible(False) # to turn off individual borders 
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['bottom'].set_linewidth(1)
            axs[i, j].spines['left'].set_linewidth(1)    
            if describe == True:
                des_name = des_list[counter]
                axs[i, j].set_title(f'{name} {des_name}', pad=-5, y=1.001)
            else:
                axs[i, j].set_title(f'{name}',  pad=-5, y=1.001)  # pad does not work on bottom plots, unless y=1.001 workaround
            
            # needs work
            # ax.legend([line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28'], loc='upper right', frameon=False) 
            [[line2, line3, line4, line5, line1], ['E19', 'P02', 'P10', 'P20', 'P28']]
                               
            counter += 1
           
    plt.suptitle(f"{title}") 
    plt.savefig(f'Traces_{title} {tissue}.png', format = 'png', bbox_inches = 'tight')   
    plt.savefig(f'Traces_{title} {tissue}.svg', format = 'svg', bbox_inches = 'tight')  
    plt.show()
    plt.clf()
    plt.close()


# Traces rhythmic AAs in SCN E19 - this is identical to eJTK + ANOVA, so do not update the Fig. 5
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Class'])
plot_gene_group_select_tissue_cc(data, name_list, 'Rhythmic Amino acid in E19 SCN acc to CircComp', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# Fig. S5: Traces rhythmic AAs in PLS P28
# This is AAs rhythmic in E19 SCN - how they look in E19-P28 PLS - use for Fig. S5
name_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name'])
des_list = list(data.loc[(data['Class'] == 'Amino acid') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Class'])
plot_gene_group_select_tissue_cc(data, name_list, 'Amino acid in PLS that are rhythmic in E19 SCN acc to CircComp', tissue='PLS', annotation=True, describe=True, des_list=des_list, y_norm=True, fnt=15, ano2w=False)


# SCN and PLS lipids for Fig. 6 UPDATED with CC for P02-P20
# CARs
name_list = list(data.loc[(data['Class'] == 'CAR') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name'])
des_list = list(data.loc[(data['Class'] == 'CAR') & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Class'])
plot_gene_group_select_tissue_cc(data, name_list, 'Rhythmic CAR in E19 SCN', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)
plot_gene_group_select_tissue_cc(data, name_list, 'Rhythmic CAR in E19 PLS', tissue='PLS', annotation=False, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# CAR NAOrns - manually
name_list = ['CAR 2:0', 'CAR 4:0', 'CAR 4:0;O', 'CAR 5:0', 'CAR 16:0', 'CAR 18:0', 'CAR 18:1', 'CAR 20:1', 'CAR 20:4', 'NAOrn 27:1;O', 'NAOrn 19:0;O']
data_Ar = data.loc[(data['Name'] == name_list[0]) | (data['Name'] == name_list[1]) | (data['Name'] == name_list[2]) | (data['Name'] == name_list[3]) | (data['Name'] == name_list[4]) | 
                   (data['Name'] == name_list[5]) | (data['Name'] == name_list[6]) | (data['Name'] == name_list[7]) | (data['Name'] == name_list[8]) | (data['Name'] == name_list[9]) | 
                   (data['Name'] == name_list[10])].loc[(data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                                          | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))]
data_Br = data_Ar
plot_gene_group_select_tissue_cc(data, name_list, 'CARS NAORn in E19 SCN acc to CircComp', tissue='SCN', annotation=False, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)
plot_gene_group_select_tissue_cc(data, name_list, 'CARS NAORn in PLS rhythmic E19 SCN acc to CircComp', tissue='PLS', annotation=False, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)

# Bases - identical in eJTK+CC as in eJTK+ANOVA
name_list = list(data.loc[((data['Class'] == 'Base') | (data['Class'] == 'Phosphate')) & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Name'])
des_list = list(data.loc[((data['Class'] == 'Base') | (data['Class'] == 'Phosphate')) & ((data['E19_emp p BH Corrected'] < bestq) & ((data['CircaComp_Mesor_Pval_E19xP28'] < bestq)
                                       | (data['CircaComp_Amp_Pval_E19xP28'] < bestq) | (data['CircaComp_Phase_Pval_E19xP28'] < bestq))), 'Class'])
plot_gene_group_select_tissue_new(data, name_list, 'Rhythmic Base in E19 SCN acc to CC', tissue='SCN', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)
plot_gene_group_select_tissue_review(data, name_list, 'Rhythmic Base in E19 PLS acc to CC', tissue='PLS', annotation=True, describe=False, des_list=des_list, y_norm=True, fnt=15, ano2w=False)

# identical
data_Ar = data.loc[(data['Name'] == name_list[0]) | (data['Name'] == name_list[1]) | (data['Name'] == name_list[2]) | (data['Name'] == name_list[3]) | 
                   (data['Name'] == name_list[4]) | (data['Name'] == name_list[5])].loc[(data['E19_emp p BH Corrected'] < bestq) & (data['1w_ANOVA_E19'] < bestq)]
data_Br = data_Ar
polar_histogram_dual(data_Ar, data_Br, ['E19_', 'P28_'], 'Q_VALUE', 'LAG', 'SCN', mydir)
polar_histogram_dual(data_Ar, data_Br, ['E19_PLS_', 'P28_PLS_'], 'Q_VALUE', 'LAG', 'PLS', mydir)


# review round 2
# polar plot of cosinor phase
#########################################################################
####### Single Polar Phase Plot #########################################
#########################################################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# mydir = 'C:\\Users\\martin.Sladek\\ownCloud\\vysledky\\_Cajka ontogenesis atlas\\SCN ONTO 2024'

gene_from_cosinor = 'Bmal1'
copy_from_prism_table = {
    "age": ["E19", "P02", "P10", "P20", "P28"],
    "phase": [0.1108, 4.173, 17.9, 18.909, 17.7],
    "sem": [3.702, 1.52, 0.2631, 0.3963, 0.5693],
    "n": [30, 30, 27, 30, 31],
    "sd": [20.28, 8.33, 1.37, 2.17, 3.17]
}

data_filt = pd.DataFrame(copy_from_prism_table)
phaseseries = data_filt['phase'].values.flatten()           # plot Phase                                   
phase_semseries = data_filt['sem'].values.flatten()                  # plot sem as width
phase_sdseries = data_filt['sd'].values.flatten()
ages_leg = data_filt['age'].values.flatten()
import matplotlib.colors as mcolors
# Convert to RGB array (values between 0 and 1)
colorcode = np.array([mcolors.to_rgb(c) for c in ['slateblue', 'grey', 'green', 'orange', 'tomato']])

# LENGTH (AMPLITUDE)
amp = np.array([1, 1, 1, 1, 1])                    # plot filtered Amplitude as length


# POSITION (PHASE)
phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
#phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
phase_sd = [polarphase(i) for i in phase_semseries]                 # if using SEM of phase, which is in hours
# phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using SD of phase, which is in hours
#phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 

# BAR PLOT VERSION
ax = plt.subplot(111, projection='polar')                                                       #plot with polar projection
bars = ax.bar(phase, amp, width=phase_sd, color=colorcode, bottom=0.9, alpha=0.8)       #transparency-> alpha=0.5, , rasterized = True, bottom=0.0 to start at center, bottom=amp.max()/3 to start in 1/3 circle
ax.set_yticklabels([])          # this deletes radial ticks
ax.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax.set_theta_direction(-1)      #reverse direction of theta increases
ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
ax.legend(bars, ages_leg, fontsize=8, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots 
ax.set_xlabel("Circadian phase (h)", fontsize=12)
# plt.title(gene_from_cosinor, fontsize=14, fontstyle='italic', pad=20)    
ax.yaxis.grid(False)   # turns off circles
ax.xaxis.grid(False)  # turns off radial grids
ax.tick_params(pad=-2)   # moves labels closer or further away from subplots

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Phase plot.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Phase plot.png', bbox_inches = 'tight')
plt.show()
plt.clf()
plt.close()


