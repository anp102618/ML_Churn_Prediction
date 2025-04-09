import pandas as pd 
import numpy as np
from Common_Utils import logger,CustomException,track_performance
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from Common_Utils import load_yaml,extract_zip,transfer_file
from Common_Utils.file_operations import FileReader
from Model_Utils.Data_Preprocessing.feature_outlier_detection import FeatureOutlierDetection
from Model_Utils.Data_Preprocessing.feature_outlier_handling import FeatureOutlierTreatment
from Model_Utils.Data_Preprocessing.feature_encoding import FeatureEncodingMethods
from Model_Utils.Data_Preprocessing.feature_sampling import FeatureOverSamplingMethods
from Model_Utils.Data_Preprocessing.feature_scaling import FeatureScalingMethods
from Common_Utils.dataframe_methods import DataFrameMethods

@track_performance
def main():
    try:
        logger.info(f"Commencing DataTransformation..")

        config_path = "./config_path.yaml"

        dict_file = load_yaml(config_path)
        raw_data_path = dict_file["DataValidation"]["raw_data_path"]
        X_train_path = dict_file["DataTransformation"]["X_train_path"]
        X_test_path = dict_file["DataTransformation"]["X_test_path"]
        y_train_path = dict_file["DataTransformation"]["y_train_path"]
        y_test_path = dict_file["DataTransformation"]["y_test_path"]

        df = FileReader.read_file(file_path= raw_data_path)
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        df.drop(columns=df.columns[[0,1]], axis=1, inplace=True)
        
        df_outlier_detected, outlier_nums_col  = FeatureOutlierDetection.detect_outliers_in_df(df,'zscore',"Churn")
        df_outlier_handled = FeatureOutlierTreatment.handling_outliers_df(df,outlier_nums_col,'yeo_johnson')

        ordinal_cols = ['Card_Category', 'Education_Level', 'Income_Category']
        categories = {"Education_Level":['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
              "Income_Category":['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
              "Card_Category":['Blue', 'Silver', 'Gold', 'Platinum'],
              }
        ohe_cols = ['Gender', 'Marital_Status']
        df_feature_encoded = FeatureEncodingMethods.feature_encoded_df(df_outlier_handled,ohe_cols,'onehot') 
        df_feature_encoded = FeatureEncodingMethods.feature_encoded_df(df_feature_encoded,ordinal_cols,'ordinal', categories) 

        X_train, X_test, y_train, y_test,X,y = DataFrameMethods.split_train_test(df= df_feature_encoded, target_column= "Churn", test_size= 0.3, random_state= 42)

        X_train_resampled, y_train_resampled, df_train_resampled = FeatureOverSamplingMethods.oversampled_balanced_df(X_train, y_train, X, "Churn", "smote")

        X_train_scaled, X_test_scaled, df_train_scaled, df_test_scaled = FeatureScalingMethods.feature_scaled_df(X_train_resampled, X_test, y_train_resampled, y_test, X,"standardization")

        X_train_scaled.to_csv(os.path.join(X_train_path, "X_train.csv"),index = False)
        X_test_scaled.to_csv(os.path.join(X_test_path, "X_test.csv"),index = False)
        y_train_resampled.to_csv(os.path.join(y_train_path, "y_train.csv"),index = False)
        y_test.to_csv(os.path.join(y_test_path, "y_test.csv"),index = False)

        logger.info(f"Data Transformation succesfully complete..")

    except CustomException as ce:
        logger.error(f"Exception found: {ce}")

if __name__ == "__main__":
    main()