# Hierarchical Structure Learning For Time Series Anomaly Detection

A multi-view time series anomaly detection method based on a Hierarchical Graph Neural Network to improve the model’s capability to capture effective representations from data.

# Requirements

1. Python >= 3.9
2. Pytorch >= 2.0

# Real Dataset Results

1. Create subfolders for each dataset within 'data'. say 'data/swat' for SWaT
2. Appropriately process the test.csv file released for each dataset
3. Copy the test.csv in the respective data subfolder. say, 'data/swat/test.csv'
4. Change dataset_name='hai' if the current run is done for dataset HAI
5. Run 'MVL_Real_Data.ipynb' to collect results

# Synthetic Dataset Results

Run 'MVL_Synthetic_Data.ipynb' to collect results

NOTE : Seed can be changed to a specific value in both cases by setting the 'seed' variable in the third cell for both synthetic and real data
