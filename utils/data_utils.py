
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler


##########################################################
#************** TIME SERIES DATA PREPROCESSING **********#
##########################################################

#------------------------------------------------------------------------
def normalize(data):
 scaler = MinMaxScaler(feature_range=(0, 1))
 scaler = scaler.fit(data)
 data = scaler.transform(data)
 return data

##########################################################
#*********** GET THE SPECIFIC MULTI VIEW DATA ***********#
##########################################################

def get_MV_Data(dataset_name='hai'): 
    if dataset_name=='hai':
       return get_MV_HAI_data()
    elif dataset_name=='swat':
       return get_MV_SWaT_data()
    elif dataset_name=='wadi':
       return get_MV_WADI_data()

##########################################################
#*********** GET HAI DATA *******************************#
##########################################################

#------------------------------------------------------------------------
def get_MV_HAI_data(): 
   file_path = './data/hai/test2.csv'

   df = pd.read_csv(file_path)

   prefix = 'P1'
   df_P1 = df.filter(regex=f'^{prefix}', axis=1)
   prefix = 'P2'
   df_P2 = df.filter(regex=f'^{prefix}', axis=1) 
   prefix = 'P3'
   df_P3 = df.filter(regex=f'^{prefix}', axis=1)
   prefix = 'P4'
   df_P4 = df.filter(regex=f'^{prefix}', axis=1)
 
   df_views = [df_P1, df_P2, df_P3, df_P4]

   data_views = []

   for df_view in df_views:
      data_view = df_view.values
      data_view = np.transpose(data_view)
      data_views.append(normalize(data_view))
      print(data_view.shape)

   df_labels = df['attack']
   
   return data_views, df_labels
 
##########################################################
#*********** GET SWaT DATA *******************************#
##########################################################

#------------------------------------------------------------------------
def get_MV_SWaT_data(): 
 file_path = './data/swat/test1.csv'

 df = pd.read_csv(file_path)

 df_P1 = df[['FIT101', 'LIT101', 'MV101', 'P101', 'P102']]
 df_P2 = df[['FIT201', 'AIT201', 'AIT202', 'AIT203', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206']]
 df_P3 = df[['FIT301', 'DPIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302']]
 df_P4 = df[['FIT401', 'AIT401', 'AIT402', 'LIT401', 'UV401', 'P501', 'P402', 'P403', 'P404']]
 df_P5 = df[['FIT501', 'FIT502', 'FIT503', 'FIT504', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'PIT501', 'PIT502', 'PIT503', 'P501', 'P502']]
 df_P6 = df[['FIT601', 'P601', 'P602', 'P603']]

 df_views = [df_P1, df_P2, df_P3, df_P4, df_P5, df_P6]

 data_views = []

 for df_view in df_views:
  df_view = df_view.apply(pd.to_numeric, errors='coerce')
  
  has_nan_values = df_view.isna().any().any()
  print(has_nan_values)
  
  if has_nan_values:
     df_view = df_view.fillna(df_view.mean())
  
  data_view = df_view.values
  data_view = np.transpose(data_view)
  data_views.append(normalize(data_view))
  print(data_view.shape)

 df_labels = df['attack']
 
 return data_views, df_labels

##########################################################
#*********** GET WADI DATA *******************************#
##########################################################

#------------------------------------------------------------------------
def get_MV_WADI_data(): 
   file_path = './data/wadi/test.csv'

   df = pd.read_csv(file_path)

   prefix = '1_'
   df_P1 = df.filter(regex=f'^{prefix}', axis=1)
   prefix = '2_'
   df_P2 = df.filter(regex=f'^{prefix}', axis=1) 
   prefix = '3_'
   df_P3 = df.filter(regex=f'^{prefix}', axis=1)
 
   df_views = [df_P1, df_P2, df_P3]

   data_views = []

   for df_view in df_views:
     
      df_view = df_view.apply(pd.to_numeric, errors='coerce')
      
      has_nan_values = df_view.isna().any().any()
      print(has_nan_values)
      
      if has_nan_values:
         df_view = df_view.fillna(df_view.mean())
         
      data_view = df_view.values
      data_view = np.transpose(data_view)
      data_views.append(normalize(data_view))
      print(data_view.shape)

   df_labels = df['attack']
   
   return data_views, df_labels
