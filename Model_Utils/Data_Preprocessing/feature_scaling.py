import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler
from abc import ABC, abstractmethod
from Common_Utils import logger, CustomException, track_performance

# Strategy Interface
class ScalingStrategy(ABC):
    @abstractmethod
    def scale(self, X_train, X_test, y_train, y_test,feature_df):
        pass

# Concrete Strategies
class MinMaxScaling(ScalingStrategy):
    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_df.columns)
            logger.info("Min-Max Scaling applied successfully.")
            df_scaled_train = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
            df_scaled_test = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)
            return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
        except CustomException as e:
            logger.error(f"Error in Min-Max Scaling: {e}")
            

class Standardization(ScalingStrategy):
    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_df.columns)
            logger.info("Standardization applied successfully.")
            df_scaled_train = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
            df_scaled_test = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)
            joblib.dump(scaler, "./Fast_API/preprocessors/scaler.joblib")
            return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
        except CustomException as e:
            logger.error(f"Error in Standardization: {e}")
            

class RobustScaling(ScalingStrategy):
    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_df.columns)
            logger.info("Robust Scaling applied successfully.")
            df_scaled_train = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
            df_scaled_test = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)
            return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
        except CustomException as e:
            logger.error(f"Error in Robust Scaling: {e}")
            

class Normalization(ScalingStrategy):
    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            scaler = Normalizer()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_df.columns)
            logger.info("Normalization applied successfully.")
            df_scaled_train = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
            df_scaled_test = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)
            return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
        except CustomException as e:
            logger.error(f"Error in Normalization: {e}")
            

class MaxAbsScaling(ScalingStrategy):
    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            scaler = MaxAbsScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_df.columns)
            X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_df.columns)
            logger.info("MaxAbs Scaling applied successfully.")
            df_scaled_train = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
            df_scaled_test = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)
            return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
        except CustomException as e:
            logger.error(f"Error in MaxAbs Scaling: {e}")
            
# Context Class
class DataFrameScaler:
    def __init__(self, strategy: ScalingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ScalingStrategy):
        self.strategy = strategy

    def scale(self, X_train, X_test, y_train, y_test, feature_df):
        try:
            return self.strategy.scale(X_train, X_test, y_train, y_test, feature_df)
        except CustomException as e:
            logger.error(f"Error in scaling: {e}")
            

#class ScalingMethods:
class FeatureScalingMethods:
    def __init__(self):
        pass

    methods = {
        "minmax_scaling":MinMaxScaling(),
        "standardization":Standardization(),
        "normalization":Normalization(),
        "robust_scaling":RobustScaling(),
        "maxabs_scaling":MaxAbsScaling(),
    }

    @track_performance
    @staticmethod
    def feature_scaled_df(X_train, X_test, y_train, y_test, feature_df,chosen_strategy:str):
        try:
            if chosen_strategy in FeatureScalingMethods.methods:
                data_scaled = DataFrameScaler(strategy=FeatureScalingMethods.methods[chosen_strategy])
                X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test = data_scaled.scale(X_train, X_test, y_train, y_test, feature_df)
                return X_train_scaled_df, X_test_scaled_df, df_scaled_train, df_scaled_test
            
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")
