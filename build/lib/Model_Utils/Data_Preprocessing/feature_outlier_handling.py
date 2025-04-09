import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import yeojohnson
from sklearn.preprocessing import PowerTransformer
from Common_Utils import logger, CustomException, track_performance


 #Strategy Interface
class OutlierHandlerStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

# Concrete Strategy: Remove Outliers using IQR
class RemoveOutliersIQR(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for column in columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                original_count = df.shape[0]
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
                new_count = df.shape[0]
                logger.info(f'Removed outliers from {column} using IQR. Rows before: {original_count}, after: {new_count}.')
            return df
        except CustomException as e:
            logger.error(f'Error removing outliers using IQR: {e}')
           

# Concrete Strategy: Cap Outliers using Z-Score
class CapOutliersZScore(OutlierHandlerStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for column in columns:
                mean = df[column].mean()
                std = df[column].std()
                z_scores = (df[column] - mean) / std
                df[column] = np.where(z_scores > self.threshold, mean + self.threshold * std,
                                      np.where(z_scores < -self.threshold, mean - self.threshold * std, df[column]))
                logger.info(f'Capped outliers in {column} using Z-Score with threshold {self.threshold}.')
            return df
        except CustomException as e:
            logger.error(f'Error capping outliers using Z-Score: {e}')
        


# Concrete Strategy: Log Transformation
class LogTransformation(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for column in columns:
                min_value = df[column].min()
                if min_value <= 0:
                    shift_value = abs(min_value) + 1
                    df[column] = np.log(df[column] + shift_value)
                    logger.info(f'Applied Log Transformation with shift to {column}. Shift value: {shift_value}.')
                else:
                    df[column] = np.log(df[column])
                    logger.info(f'Applied Log Transformation to {column} without shift.')
            return df
    
        except CustomException as e:
            logger.error(f'Error applying Log Transformation: {e}')
        


# Concrete Strategy: Yeo-Johnson Transformation
class YeoJohnsonTransformation(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            for column in columns:
                df[column] = pt.fit_transform(df[[column]])
                logger.info(f'Applied Yeo-Johnson Transformation to {column}.')
            return df
        except Exception as e:
            logger.error(f'Error applying Yeo-Johnson Transformation: {e}')
            return df


# Concrete Strategy: Mean Imputation
class MeanImputation(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for column in columns:
                mean_value = df[column].mean()
                df[column] = np.where(self._is_outlier(df[column]), mean_value, df[column])
                logger.info(f'Applied Mean Imputation to {column}.')
            return df
        except CustomException as e:
            logger.error(f'Error applying Mean Imputation: {e}')
            

    def _is_outlier(self, series: pd.Series) -> pd.Series:
        return np.abs(series - series.mean()) > 3 * series.std()

# Concrete Strategy: Median Imputation
class MedianImputation(OutlierHandlerStrategy):
    def handle(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for column in columns:
                median_value = df[column].median()
                df[column] = np.where(self._is_outlier(df[column]), median_value, df[column])
                logger.info(f'Applied Median Imputation to {column}.')
            return df
        except CustomException as e:
            logger.error(f'Error applying Median Imputation: {e}')

    def _is_outlier(self, series: pd.Series) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))

# Context Class
class OutlierContext:
    def __init__(self, strategy: OutlierHandlerStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierHandlerStrategy):
        self.strategy = strategy

    def apply_strategy(self, df: pd.DataFrame, column: list) -> pd.DataFrame:
        return self.strategy.handle(df, column)

class FeatureOutlierTreatment:
    def __init__(self):
        pass

    methods = {
        "remove_outlier_iqr":RemoveOutliersIQR(),
        "cap_outlier_zscore":CapOutliersZScore(threshold=3),
        "log_transform":LogTransformation(),
        "yeo_johnson":YeoJohnsonTransformation(),
        "mean":MeanImputation(),
        "median":MedianImputation(),

    }

    @track_performance
    @staticmethod
    def handling_outliers_df(df:pd.DataFrame,cols:list,chosen_strategy:str) -> pd.DataFrame:
        try:

            if chosen_strategy in FeatureOutlierTreatment.methods:
                Outlier_transform=OutlierContext(strategy= FeatureOutlierTreatment.methods[chosen_strategy])
                df_transformed = Outlier_transform.apply_strategy(df, cols)
                return df_transformed
            
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")
