import os
import scipy.io
import numpy as np

# Specify the path to the folder
folder_path = 'datapackage'

# Specify the keys you want to combine
selected_keys = ['DOY', 'PA', 'Prcp', 'Prcp', 'Tdew', 'Temp', 'Tmax', 'Tmin', 'Wind', 'HWFlag', 'Tref' ] #'StationLat', 'StationLon', 

# List all files in the folder
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Filter only .mat files
mat_files = [f for f in files if f.endswith('.mat')]

index = 1

# Access and load data from each .mat file
for mat_file in mat_files:
    mat_file_path = os.path.join(folder_path, mat_file)
    
    # Load data from the .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # Combine the selected arrays into a single array
    selected_arrays = [mat_data[key] for key in selected_keys]
    combined_array = np.hstack(selected_arrays)
    
    csv_file_path = 'csv/'+str(index)+'.csv'

    # Save the NumPy array to a CSV file
    np.savetxt(csv_file_path, combined_array.T, delimiter=',')
    
    index = index + 1
