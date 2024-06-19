
import pandas as pd
import numpy as np

train_file_path = '../../../DATASETS/HAI/train.csv'
downsampled_train_file_path = '../../../DATASETS/HAI/downsampled_train.csv'
test_file_path = '../../../DATASETS/HAI/test.csv'
columns_file_path = '../../../DATASETS/HAI/list.txt'

#--------------------- Get the list of columns -------------------
def get_columns():
  df = pd.read_csv(downsampled_train_file_path) 
  print('HAI Columns : ',df.columns) 
  
  # Extract column names as a list
  column_names = df.columns.tolist()

  # Write the column names to the text file
  with open(columns_file_path, 'w') as file:
    file.write("\n".join(column_names))

  print(f"Column names saved to {columns_file_path}")
  
#--------------------- Get the list of columns -------------------
def print_filter_columns():
  df = pd.read_csv(downsampled_train_file_path) 
 
  # Specify the expression you want to use as a prefix
  prefix_expression = 'P1'

  # Get column names starting with the specified prefix
  filtered_columns = df.filter(regex=f'^{prefix_expression}', axis=1)

  # Display the result
  print(filtered_columns)
  
#--------------------- Get downsampled train data -------------------
def get_downsampled_train_data():
 
 df = pd.read_csv(train_file_path, delimiter=';') 
 print('HAI Train Original Data Shape : ',df.shape) 
 print(df.head())
 
 #------ Removing white spaces from column names
 df = df.rename(columns=lambda x: x.strip())
 
 df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
 df = df.set_index(['time'])
 
 #------ Checking the presence of nan values and replacing if any
 has_nan_values = df.isna().any().any()
 
 if has_nan_values:
  df = df.fillna(df.mean())
  
 #------ Converting values to numeric
 df = df.apply(pd.to_numeric, errors='coerce')
 
 #------ Performing downsampling
 #To specify the downsampling frequency when using pandas to downsample a time series dataset, you can pass the desired frequency as an argument to the resample() function. 
 #The frequency is specified using a string with a combination of a numeric value and a time unit. 
 #Here are some common time units you can use - 'D': Day; 'W': Week; 'M': Month; 'Q': Quarter; 'A': Year; 'H' : hours; 'T' : minutes; 'S' for seconds
 #Additionally, you can combine numeric values with time units to create custom intervals. For example, '2W' represents a two-week interval.
 downsampling_frequency = '10S'
 downsampled_df = df.resample(downsampling_frequency).mean()
 
 #------ Printing final rows
 print(downsampled_df.head())
 
 #------ Print shape of downsampled df
 print('Downsampled Train Data Shape : ',downsampled_df.shape)
 
 downsampled_df.to_csv('../../../DATASETS/HAI/downsampled_train.csv')
 
#--------------------- Get downsampled test data -------------------
def get_downsampled_test_data():
 
 df = pd.read_csv(test_file_path, delimiter=';') 
 print('HAI Test Original Data Shape : ',df.shape) 
 print(df.head())
 
 #------ Removing white spaces from column names
 df = df.rename(columns=lambda x: x.strip())
 
 df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
 df = df.set_index(['time'])
 
 #------ Checking the presence of nan values and replacing if any
 has_nan_values = df.isna().any().any()
 
 if has_nan_values:
  df = df.fillna(df.mean())
  
 #------ Converting values to numeric
 df = df.apply(pd.to_numeric, errors='coerce')
 
 #------ Performing downsampling
 #To specify the downsampling frequency when using pandas to downsample a time series dataset, you can pass the desired frequency as an argument to the resample() function. 
 #The frequency is specified using a string with a combination of a numeric value and a time unit. 
 #Here are some common time units you can use - 'D': Day; 'W': Week; 'M': Month; 'Q': Quarter; 'A': Year; 'H' : hours; 'T' : minutes; 'S' for seconds
 #Additionally, you can combine numeric values with time units to create custom intervals. For example, '2W' represents a two-week interval.
 downsampling_frequency = '10S'
 downsampled_df = df.resample(downsampling_frequency).mean()
 
 #------ Printing final rows
 print(downsampled_df.head())
 
 #------ Print shape of downsampled df
 print('Downsampled Test Data Shape : ',downsampled_df.shape)
 
 downsampled_df.to_csv('../../../DATASETS/HAI/downsampled_test.csv')
 
#get_downsampled_train_data()
#get_downsampled_test_data()
#get_columns()
print_filter_columns()
