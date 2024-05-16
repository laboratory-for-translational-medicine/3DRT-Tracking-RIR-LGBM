#Importing modules
import os
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sktime.forecasting.compose import make_reduction
import csv
# import time
# all_times = []

# Preparing
directory = r"C:\Users\elodi\Desktop\LightGBM_Displacements"
prediction_horizon = 3 # How many measurements are we predicting
window_size = 5 # How many measurements are we using
training_size = 60 # We will first use 60 measurements for training the model
nbr_file_process = 10 # Can change this to speed up development of algorithms

params = {
    "max_depth": 10,
    "num_leaves": 16,
    "learning_rate": 0.02,
    "n_estimators": 300
}

nb_file_processed = 0
warnings.filterwarnings("ignore") # Preventing warning to be displayed

regressorX = lgb.LGBMRegressor(**params)
regressorY = lgb.LGBMRegressor(**params)
regressorZ = lgb.LGBMRegressor(**params)
forecasterX = make_reduction(regressorX, window_length=window_size, strategy="recursive")
forecasterY = make_reduction(regressorY, window_length=window_size, strategy="recursive")
forecasterZ = make_reduction(regressorZ, window_length=window_size, strategy="recursive")
fh = np.arange(prediction_horizon) + 1

# Loop through each file in the directory
allTheFoldersInDirectory = os.listdir(directory)
for folders in allTheFoldersInDirectory:
    allTheFilenamesInDirectory = os.listdir(directory+ '\\'+ folders)
    for filename in allTheFilenamesInDirectory:
        if nbr_file_process > nb_file_processed and 'Measurement' in filename: #taking only measurement files
            print(filename)
        
            #Reading the data file and extracting the measurements 
            df_full = pd.read_csv(os.path.join(directory, os.path.join(folders,filename))) #reading the file
            df_full.drop(['Image', 'Registration Status'], axis=1, inplace=True) # Extracting only the displacement measurements
            df = df_full.iloc[::2] #taking only 1 measurement out of 2 for X and Y
            
            dfX = pd.DataFrame(df['X'], columns=['X'])
            dfY = pd.DataFrame(df['Y'], columns=['Y'])
            dfZ = pd.DataFrame(df_full['Z'], columns=['Z'])
            #Doubling the training set
            trainX = pd.concat([dfX[dfX.index<training_size],dfX[dfX.index<training_size]], ignore_index=True).values.reshape(-1, 1)
            testX = dfX[dfX.index>=training_size].values.reshape(-1, 1)
            trainY = pd.concat([dfY[dfY.index<training_size],dfY[dfY.index<training_size]], ignore_index=True).values.reshape(-1, 1)
            testY = dfY[dfY.index>=training_size].values.reshape(-1, 1)
            trainZ = pd.concat([dfZ[dfZ.index<training_size],dfZ[dfZ.index<training_size]], ignore_index=True).values.reshape(-1, 1)
            # Instead of taking the measurements Z right away, we take the average of the last 2 ones
            trainZ = (trainZ[:-1]+trainZ[1:])/2
            testZ = dfZ[dfZ.index>=training_size-1].values.reshape(-1, 1)
            testZ = (testZ[:-1]+testZ[1:])/2

            forecasterX.fit(trainX)
            predX = forecasterX.predict(fh=fh)
            forecasterY.fit(trainY)
            predY = forecasterY.predict(fh=fh)
            forecasterZ.fit(trainZ)
            predZ = forecasterZ.predict(fh=fh)
            all_preds = []
            all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                              predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                              predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
            
            # orientation image is now 'coronal'
            while len(testZ) != 0: # Retraining when new measurement is available 
                
                # Y was just measured
                # start_time = time.time()
                trainY = np.append(trainY, testY[0])
                testY = np.delete(testY,0)
                forecasterY.fit(trainY)
                predY = forecasterY.predict(fh=fh)
                # end_time = time.time()
                # all_times.append(end_time - start_time)
                
                # start_time = time.time()
                trainZ = np.append(trainZ, testZ[0])
                testZ = np.delete(testZ,0)
                forecasterZ.fit(trainZ)
                predZ = forecasterZ.predict(fh=fh)
                # end_time = time.time()
                # all_times.append(end_time - start_time)
                
                all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                                  predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                                  predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
                # Check we are not done with the data
                if len(testZ)==0:
                    break
                # X was just measured
                # start_time = time.time()
                trainX = np.append(trainX, testX[0])
                testX = np.delete(testX,0)
                forecasterX.fit(trainX)
                predX = forecasterX.predict(fh=fh)
                # end_time = time.time()
                # all_times.append(end_time - start_time)
                
                # start_time = time.time()
                trainZ = np.append(trainZ, testZ[0])
                testZ = np.delete(testZ,0)
                forecasterZ.fit(trainZ)
                predZ = forecasterZ.predict(fh=fh)    
                # end_time = time.time()
                # all_times.append(end_time - start_time)
                
                all_preds.append([predX[0][0], predY[0][0], predZ[0][0],
                                  predX[1][0], predY[1][0], predZ[1][0],  ## Hard-coded for now. To be improved. 
                                  predX[2][0], predY[2][0], predZ[2][0]]) ## We are reporting the 3 forecasted measurements
            
            filepath_pred_data = directory+ '\\'+ folders + '\\Prediction_'+ '_'.join(filename.split('_')[1:-1]) +'.csv'      
            # Writing data to CSV file
            with open(filepath_pred_data, 'w', newline='') as csvPreds:
                csvPreds_writer = csv.writer(csvPreds)
                csvPreds_writer.writerow(['X', 'Y', 'Z', 'X+2', 'Y+2', 'Z+2', 'X+3', 'Y+3', 'Z+3']) # Write headers ## Hard-coded for now. To be improved. 
                csvPreds_writer.writerows(all_preds) # Write data from the lists
            
            nb_file_processed += 1  
            
# print(np.average(all_times))
            