import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from math import log
from scipy.stats import linregress
import ast
import statsmodels.stats.multitest as multi
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import plotly.express as px

from matplotlib_venn import venn3
import holoviews as hv
from holoviews import opts, dim
import holoviews.plotting.mpl
import itertools as it

def read_raw_protein_file(proteingroupfile):
    filters =  ['Reverse', 'Only identified by site']
    data = pd.read_csv(proteingroupfile, sep='\t', na_values=['0'], low_memory=False)
    data = data[data[filters].isnull().all(1)]
    
    out = data[data['Potential contaminant'].notnull()]
    out_cast = out['Protein IDs'].tolist()
    to_keep = data[data['Gene names']=='ALB']
    out_cast = [x for x in out_cast if x not in list(to_keep['Protein IDs'])]
    data = data[~data['Protein IDs'].isin(out_cast)]
    filters.append('Potential contaminant')
    data = data.drop(filters, axis=1)
    data = data.reset_index(drop=True)
    return data

def extract_datamatrix(ProteinGroups, index='Majority protein IDs'):
    df=read_raw_protein_file(ProteinGroups)
    experiment_Cols = [col for col in df if col.startswith('LFQ intensity')]
    protein_Cols = [index]
    Cols = experiment_Cols + protein_Cols
    df=df[Cols]
    df['Majority protein IDs']=df['Majority protein IDs'].str.split(';').str[0]
    df=df.set_index('Majority protein IDs')
    
    return df

def rename_experiment (data, base_label, new_label, labelfile):
    label_dict=dict(zip(labelfile[base_label],labelfile[new_label]))
    data=data.rename(columns=label_dict)
    return data

def extract_datamatrix_DIA(Report, labelfile):
    data = Report.copy()
    for i in ['PG.ProteinAccessions', 'PG.Genes']:
        data[i]=data[i].str.split(';').str[0]
    data=data.rename({'PG.Genes': 'Gene names', 'PG.ProteinAccessions': 'Protein ID'}, axis=1)
    data=rename_experiment(data, base_label='File name', new_label='Sample ID', labelfile=labelfile)
    data=data.replace({'Filtered': np.float64('NaN')})
    return data

def extract_datamatrix_copynumber(ProteinGroups, index='Majority protein IDs'):
    df=read_raw_protein_file(ProteinGroups)
    experiment_Cols = [col for col in df if col.startswith('Copy number')]
    protein_Cols = [index]
    Cols = experiment_Cols + protein_Cols
    df=df[Cols]
    df['Majority protein IDs']=df['Majority protein IDs'].str.split(';').str[0]
    df=df.set_index('Majority protein IDs')
    
    return df

def extract_Parameters(ProteinGroups, index='Majority protein IDs'):
    df=read_raw_protein_file(ProteinGroups)
    experiment_Cols = [col for col in df if col.startswith('LFQ intensity')]
    SequenceCoverage_Cols = [col for col in df if col.startswith('Sequence coverage')]
    Peptide_Cols = [col for col in df if col.startswith('Peptides')]
    Uniquepeptide_Cols = [col for col in df if col.startswith('Unique peptides')]
    protein_Cols = [index]
    protein_per_group = []
    for i in np.arange(df.shape[0]):
        count=len(df['Majority protein IDs'][i].split(';'))
        protein_per_group.append(count)
    
    Cols = protein_Cols + SequenceCoverage_Cols + Peptide_Cols +Uniquepeptide_Cols + ['Mol. weight [kDa]']
    df=df[Cols]
    df['Proteins_per_group'] = protein_per_group
    df['Majority protein IDs']=df['Majority protein IDs'].str.split(';').str[0]
    df=df.set_index('Majority protein IDs')
    
    return df

def imputation_normal_distribution(df):
    data_imputed = df.copy()
    for i in data_imputed.loc[:, data_imputed.isnull().any()]:
        missing = data_imputed[i].isnull()
        std = data_imputed[i].std()
        mean = data_imputed[i].mean()
        sigma = std*0.3
        mu = mean - (std*1.8)
        data_imputed.loc[missing, i] = np.random.normal(mu, sigma, size=len(data_imputed[missing]))
        
    return data_imputed

def twosample_ttest (data, grouping):
    df=data.copy()
    df=df.T.reset_index().set_index('Samples')
    df['group']=df.index.map(grouping.get)
    group_list=[str(x) for x in input().split()]
    df_anova=df[df['group'].isin(group_list)]
    df_anova=df_anova.reset_index().set_index(['group', 'Samples']).T
    columns = ['protein', 't-statistics', 'pvalue', '-Log pvalue', 'Log difference']
    scores= []    
    #'index'as 'Protein ID'
    for i in np.arange(len(df_anova.index)):
        groups=[]
        for j in group_list:
            groups.append(list(df_anova[j].iloc[i,:]))
        mean_group1=np.mean(groups[0])
        mean_group2=np.mean(groups[1])
        log10_difference=mean_group2-mean_group1
        
        t_val, p_val = stats.ttest_ind(*groups)
        log = -math.log(p_val, 10)
        scores.append((df_anova.index[i], t_val, p_val, log, log10_difference))        
    scores = pd.DataFrame(scores)
    scores.columns = columns
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['pvalue'], alpha=0.1, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    
    return scores  

def filter_valid_values (data, mapper, groups, how='any', completeness=0.7):
    """ if how is 'any', keep rows with at least one group that has completeness requirment met
        if how is 'all, keep rows that all groups must meet completeness requirement
        """
    data_new = data.copy()
    df_new=data.rename(mapper=mapper, axis=1)
    df = pd.DataFrame(data.index)
    for i in groups:
        valid_values= []
        percentage = []
        for j in np.arange(df_new.shape[0]):
            count = df_new[i].loc[j].count()
            per = count/df_new[i].shape[1]
            percentage.append(per)

        df['%valid values'+i] = percentage

    counts_below=[]
    for i in np.arange(df.shape[0]):
        count_below=sum(list(df.loc[i]<completeness))
        counts_below.append(count_below)
    data_new['count_below']=counts_below
    
    if how=='any':
        data_new = data_new[data_new.count_below != len(groups)]
    elif how=='all':
        data_new = data_new[data_new.count_below == 0]
    data_new.drop('count_below', axis=1, inplace=True)
    
    return data_new

def tidy_split(df, column, sep='|', keep=False):
    """
    Split the values of a column and expand so the new DataFrame has one split
    value per row. Filters rows where the column is missing.

    Params
    ------
    df : pandas.DataFrame
        dataframe with the column to split and expand
    column : str
        the column to split and expand
    sep : str
        the string used to split the column's values
    keep : bool
        whether to retain the presplit value as it's own row

    Returns
    -------
    pandas.DataFrame
        Returns a dataframe with the same columns as `df`.
    """
    indexes = list()
    new_values = list()
    df = df.dropna(subset=[column])
    for i, presplit in enumerate(df[column].astype(str)):
        values = presplit.split(sep)
        if keep and len(values) > 1:
            indexes.append(i)
            new_values.append(presplit)
        for value in values:
            indexes.append(i)
            new_values.append(value)
    new_df = df.iloc[indexes, :].copy()
    new_df[column] = new_values
    return new_df

def calculate_quartile_numbers (data, column):
    df = data
    Q1= df[df[column]<0.25][column].count()
    Q2= df[df[column]<0.5][column].count()
    Q3= df[df[column]<0.75][column].count()
    Q4= df[df[column]<1][column].count()
    
    Q2=Q2-Q1
    Q3=Q3-Q2-Q1
    Q4=Q4-Q3-Q2-Q1
    return [column, Q1, Q2,Q3, Q4]

def intersection(a, b):
    return list(set(a) & set(b))
    
def return_kegg_coverage(data, dictionary, pathway_dict):
    """
    Data frame row index as gene name, column index as sample name. 
    """
    df= data.copy()
    df_new = pd.DataFrame()
    df_new['pathways'] = pathway_dict.keys()
    df_new['genes'] = pathway_dict.values()
    
    column_list = []
    genenumber_list = []
    percentage_list = []
    coverage_list = []
        
    column_genes = list(data.index)
    for j in np.arange(df_new.shape[0]):
        overlapped_genes = intersection(column_genes, df_new['genes'][j])
        number_of_overlapped_genes = len(overlapped_genes)
        rate_of_coverage = number_of_overlapped_genes/len(df_new['genes'][j])
            
        column_list.append(overlapped_genes)
        genenumber_list.append(number_of_overlapped_genes)
        coverage_list.append(rate_of_coverage)
            
    column_name_genes =  'covered genes' + 'all'
    column_name_number = 'no of covered genes' + 'all'
    column_name_percentage = 'percentage of covered genes' + 'all'
        
    df_new[column_name_genes] = column_list
    df_new[column_name_number] = genenumber_list
    df_new[column_name_percentage] = coverage_list    

    
    for i in np.arange(len(data.columns)):
        column_list = []
        genenumber_list = []
        percentage_list = []
        coverage_list = []
        
        column_genes = list(data[data.columns[i]].dropna().index)
        for j in np.arange(df_new.shape[0]):
            overlapped_genes = intersection(column_genes, df_new['genes'][j])
            number_of_overlapped_genes = len(overlapped_genes)
            rate_of_coverage = number_of_overlapped_genes/len(df_new['genes'][j])
            
            column_list.append(overlapped_genes)
            genenumber_list.append(number_of_overlapped_genes)
            coverage_list.append(rate_of_coverage)
            
        column_name_genes =  'covered genes' + df.columns[i]
        column_name_number = 'no of covered genes' + df.columns[i]
        column_name_percentage = 'percentage of covered genes' + df.columns[i]
        
        df_new[column_name_genes] = column_list
        df_new[column_name_number] = genenumber_list
        df_new[column_name_percentage] = coverage_list
        
    return df_new

def calculate_pathway_percentage (pathway_dictionary, data):
    """
    Calculates and returns the fraction of a particular pathway (could also be a protein class) against entire proteome.
    'data' as pandas dataframe, row index as gene name, columns as samples.
    'pathway_dictionary' should be dictionary, keys of name of protein class, values of gene names associated with this protein class.
    """
    data = data.copy()
    cols = data.columns
    df_new = pd.DataFrame()
    df_new['Samples'] = cols
    pathways = list(pathway_dictionary.keys())
    for pathway in pathways:
        geneset = pathway_dictionary[pathway]
        geneset = list(set(geneset) & set(data.index))
        fraction = []
        for col in cols:
            sum_pathway = data.loc[geneset][col].sum()
            sum_col = data[col].sum()
            frac = sum_pathway/sum_col
            fraction.append(frac)
        df_new[pathway] = fraction
    df_new = df_new.set_index('Samples')
    df_new = df_new.rename_axis(columns = 'Pathways')
    return df_new

def anova_oneway (data, dv, between, detailed = False, fdr = 0.05):
    """
    One-way ANOVA (pg.anova), long data format, multiple hypothesis testing corrected by Benjamini-Hochberg.
    Note that column name of dv shouldn't contain '\t'
    More refer to: https://pingouin-stats.org/generated/pingouin.anova.html
    """
    columns = ['protein', 'Source', 'SS', 'DF', 'MS', 'F', 'p-unc', 'np2']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_anova = data.loc[i]
        anova = pg.anova(data = df_anova, dv = dv, between = between)
        anova['protein'] = i 
        scores = scores.append(anova)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['p-unc']))
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    scores = scores[scores.Source != 'residual']
    
    #FDR correction
    reject, qvalue = multi.fdrcorrection(scores['p-unc'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores 

def ancova_pg (data, dv, between, covar, fdr = 0.05):
    """
    Analysis of covariance (ANCOVA) (pg.ancova), long data format, multiple hypothesis testing corrected by Benjamini-Hochberg.
    Note that column name of dv shouldn't contain '\t'
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "between": Name of column containing the between factor.
    "covar": Name(s) of column(s) containing the covariate. 
    More refer to: https://pingouin-stats.org/generated/pingouin.ancova.html
    """
    columns = ['protein', 'Source', 'SS', 'DF', 'F', 'p-unc']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_ancova = data.loc[i]
        ancova = pg.ancova(data = df_ancova, dv = dv, between = between, covar = covar)
        num_covar = len(covar)
        ancova['protein'] = i
        scores = scores.append(ancova, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['p-unc']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    scores = scores[scores.Source != 'Residual']
    
    #FDR correction
    np.random.seed(123)
    reject, qvalue = multi.fdrcorrection(scores['p-unc'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores

def pairwisetukey_pg (data, dv, between, fdr = 0.05):
    """
    This is a wrapper for pingouin.pairwise_tukey, multiple hypothesis testing corrected by Benjamini-Hochberg.
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "between": Name of column containing the between factor.
    More refer to: https://pingouin-stats.org/generated/pingouin.pairwise_tukey.html
    """
    columns = ['protein', 'A', 'B', 'mean(A)', 'mean(B)', 'diff', 'SE', 'tail', 'T', 'p-tukey', 'efsize', 'eftype']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_tukey = data.loc[i]
        tukey = pg.pairwise_tukey(data = df_tukey, dv = dv, between = between)
        tukey['protein'] = i
        scores = scores.append(tukey, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['p-tukey']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    #FDR correction
    np.random.seed(123)
    reject, qvalue = multi.fdrcorrection(scores['p-tukey'], alpha=0.05, method='indep')
    scores['qvalue'] = qvalue
    scores['rejected'] = reject
    return scores

def homoscedasticity_pg (data, dv, group):
    """
    This is a wrapper of pingouin.homoscedasticity test.
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "group": Name of column containing the between factor.
    More refer to: https://pingouin-stats.org/generated/pingouin.homoscedasticity.html
    """
    columns = ['W', 'pval', 'equal_var']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_homoscedasticity = data.loc[i]
        homoscedasticity = pg.homoscedasticity(data=df_homoscedasticity, dv=dv, group=group)
        homoscedasticity['protein'] = i
        scores = scores.append(homoscedasticity, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['pval']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    return scores

def normality_pg (data, dv, group, method='shapiro'):
    """
    This is a wrapper of pingouin.normality test.
    "data": should be long data format, with protein ID as index.
    "dv": Name of column containing the dependant variable.
    "group": Grouping factor.
    More refer to: https://pingouin-stats.org/generated/pingouin.normality.html
    """
    columns = ['index', 'W', 'pval', 'normal']
    scores = pd.DataFrame(columns = columns)
    for i in list(set(data.index)):
        df_normality = data.loc[i]
        normality = pg.normality(data=df_normality, dv=dv, group=group, method=method).reset_index()
        normality['protein'] = i
        scores = scores.append(normality, sort=False)
    scores = scores.assign(new_column = lambda x: -np.log10(scores['pval']), sort = False)
    scores = scores.rename({'new_column' : '-Log pvalue'}, axis = 1)
    
    return scores