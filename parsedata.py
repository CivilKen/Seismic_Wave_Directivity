# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:42:40 2019

@author: CivilKen
"""

import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os

currentpath = os.getcwd()
threshold = 0.5
filename = '\\ML_data.xlsx'
df = pd.read_excel(currentpath+filename)

#%% feature scaleing(standardziation)
maxmin = df[['FaultDistance', 'Depth', 'Magnitude']]
meanstd = df[['EP_distance', 'DorFrequency']]

df_maxmin = maxmin.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
df_meanstd = meanstd.apply(lambda x: (x - np.mean(x)) / np.std(x))

#%% one-hot encoding
df.activity = df.activity.map({1.0:'one', 2.0:'two'})
df.covered = df.covered.map({1.0:'cover', 0.5:'halfcover'})
ohelist = ['classification', 'activity', 'Position', 'slip_type',
           'covered', 'GeoDiv', 'Vs30Class', 'slip_direction']

## 
def OHE(df, fname):
    # OHE: one-hot encoding
    # fname: feature name
    new_col = df[fname].unique()
    new_col = new_col.astype(str) # conver nan value into string value 'nan'
    new_col = np.setdiff1d(new_col,np.array(['nan','NaN','None','none']))
    df_append = pd.DataFrame(0, index=np.arange(len(df)), columns=list(new_col))
    for i in range(len(df_append)):
        if df[fname][i] in new_col:
            df_append[df[fname][i]][i]=1
    return df_append
    
df_ohe = pd.DataFrame(index=np.arange(len(df)))
for i in ohelist:
    df_append = OHE(df,i)
    df_ohe = pd.concat([df_ohe, df_append], axis=1)
    
df_ohe.columns = ['interplate', 'intraplate', 'one', 'two', 'F', 'H', 'dip', 'strike',
       'halfcover', 'cover', 'GeoA', 'GeoB', 'GeoC', 'GeoD', 'GeoE', 'GeoG', 'Vs30A', 
       'Vs30B', 'Vs30C', 'Vs30D', 'Vs30E','left', 'normal', 'right', 'thrust']

#%% input azimuth data
input_azi = ['EP_azimuth', 'sigma1_azi', 'sigma2_azi', 'sigma3_azi', 'sigma1_plg',
             'sigma2_plg', 'sigma3_plg', 'fault_strike', 'DipDirection']

# 'iazi : input azimuth
df_iazi = pd.DataFrame(index=np.arange(len(df)))
for i in input_azi:
    df_append = pd.concat([np.cos(np.radians(df[i])), np.sin(np.radians(df[i]))], axis=1)
    df_iazi = pd.concat([df_iazi, df_append], axis=1)
    
#%% output azimuth data
ai_list = ['AI0','AI10','AI20','AI30','AI40','AI50','AI60','AI70','AI80','AI90','AI100','AI110',
           'AI120','AI130','AI140','AI150','AI160','AI170','AI180','AI190','AI200','AI210','AI220','AI230',
           'AI240','AI250','AI260','AI270','AI280','AI290','AI300','AI310','AI320','AI330','AI340','AI350',]

df_ai = pd.DataFrame(index=np.arange(len(df)))
for i in range(int(len(ai_list)/2)):
    df_ai = pd.concat([df_ai,df[ai_list[i]]+df[ai_list[i+18]]], axis=1)

# smooth-label
df_output = df_ai.apply(lambda x: x/np.linalg.norm(x), axis=1)
df_output.columns = range(18)
#df_output.columns = ['AI0','AI10','AI20','AI30','AI40','AI50','AI60','AI70','AI80',
#                     'AI90','AI100','AI110','AI120','AI130','AI140','AI150','AI160','AI170']

# arias summary
angle = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]

df_ii = df_output.apply(lambda x: np.min(x)/np.max(x), axis=1)
df_dir = df_ii.apply(lambda x: 1 if x<=threshold else 0)
df_ii = pd.DataFrame(df_ii, columns=['isotropic_index'])
df_dir = pd.DataFrame(df_dir, columns=['directivity']) 
## Arias distribution
def AriasDistribution(y):
    # y represents the results of the prediction
    # y.shape = (# of data ,18)    
    ## mirror direction
    distribution = np.zeros((y.shape[0],36))
    ariasindex = y.idxmax(axis=1)
    for i in range(y.shape[0]):
        forindex = ariasindex[i]+1
        backindex = ariasindex[i]-1
        if forindex==18:            
            if y.loc[i,0]>y.loc[i,backindex]:
                hibound = forindex
                lobound = ariasindex[i]
            else:
                hibound = ariasindex[i]
                lobound = backindex
        elif backindex==-1:
            if y.loc[i,forindex]>y.loc[i,17]:
                hibound = forindex
                lobound = ariasindex[i]
            else:
                hibound = ariasindex[i]
                lobound = backindex
        else:
            if y.loc[i,forindex]>y.loc[i,backindex]:
                hibound = forindex
                lobound = ariasindex[i]
            else:
                hibound = ariasindex[i]
                lobound = backindex
        for j in range(9):
            aidxp = hibound+j
            didxp = hibound+9+j
            aidxm = lobound-j
            didxm = lobound+9-j
            if aidxp>17:
                aidxp=aidxp-18
            elif aidxm<0:
                aidxm=aidxm+18
            distribution[i,didxp]=y.loc[i,aidxp]
            distribution[i,didxm]=y.loc[i,aidxm]        
    ## Arias Mean
    ariasmean = np.zeros((y.shape[0],1))
    for i in range(y.shape[0]):
        for j in range(36):
            ariasmean[i,0] += distribution[i,j]*(j-9)*10
    sum1=np.reshape(np.sum(distribution,axis=1),(y.shape[0],1))
    ariasmean/=sum1
    ## standard deviation
    ariasstd = np.zeros((y.shape[0],1))
    ariasvar = np.zeros((y.shape[0],1))
    ariassum = np.zeros((y.shape[0],1))
    for i in range(len(ariasmean)):
        for j in range(36):
            ariassum[i,0]+=((j-8)*10-ariasmean[i,0])**2*distribution[i,j]
        ariasvar[i,0] = (ariassum[i,0]/np.sum(distribution[i,:]))
        ariasstd[i,0] = ariasvar[i,0]**0.5        
    
    return ariasstd, ariasmean, distribution

[AriasStd, AriasMean, AriasDis] = AriasDistribution(df_output)
df_AriasStd = pd.DataFrame(AriasStd, columns=['AriasStd'])
df_AriasMean = pd.DataFrame(AriasMean, columns=['AriasMean'])

#%% write excel
df_input = pd.concat([df_maxmin, df_meanstd, df_iazi, df_ohe], axis=1)
df_summary = pd.concat([df_ii, df_dir, df_AriasStd, df_AriasMean], axis=1)

train_size = int(len(df)*0.8)
df_input_train = df_input.loc[:train_size,:]
df_output_train = df_output.loc[:train_size,:]
df_summary_train = df_summary.loc[:train_size,:]
df_input_val = df_input.loc[train_size:,:]
df_output_val = df_output.loc[train_size:,:]
df_summary_val = df_summary.loc[train_size:,:]

workbook = ExcelWriter('ML_data_FM5.xlsx')
df_input_train.to_excel(workbook, sheet_name='input_train')
df_output_train.to_excel(workbook, sheet_name='output_train') 
df_summary_train.to_excel(workbook, sheet_name='summary_train') 
df_input_val.to_excel(workbook, sheet_name='input_val') 
df_output_train.to_excel(workbook, sheet_name='output_train') 
df_summary_train.to_excel(workbook, sheet_name='summary_train')
workbook.save()