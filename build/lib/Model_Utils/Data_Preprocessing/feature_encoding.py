import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from Common_Utils import logger, CustomException, track_performance
from abc import ABC, abstractmethod



# Strategy Interface
class EncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        pass

# Concrete Strategies
class LabelEncoding(EncodingStrategy):
    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        try:
            encoder = LabelEncoder()
            for col in cols:
                df[col] = encoder.fit_transform(df[col])
                logger.info(f"Label Encoding applied successfully on column: {col}.")
            return df
        except CustomException as e:
            logger.error(f"Error in Label Encoding: {e}")
            

class OneHotEncoding(EncodingStrategy):
    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        try:
            for col in cols:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                encoded_data = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded_data, columns=[f"{col}_{cat}" for cat in encoder.categories_[0][1:]])
                df = df.drop(col, axis=1).reset_index(drop=True)
                df = pd.concat([df, encoded_df], axis=1)
                logger.info(f"One-Hot Encoding applied successfully on column: {col}.")
            return df
        except CustomException as e:
            logger.error(f"Error in One-Hot Encoding: {e}")
            

class OrdinalEncoding(EncodingStrategy):
    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        dict_ordinal = {}
        try:
            for col in cols:
                encoder = OrdinalEncoder(categories=[categories[col]])
                df[col] = encoder.fit_transform(df[[col]])
                logger.info(f"Ordinal Encoding applied successfully on column: {col}.")
            return df
        except CustomException as e:
            logger.error(f"Error in Ordinal Encoding: {e}")
            


# Context Class
class DataFrameEncoder:
    def __init__(self, strategy: EncodingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: EncodingStrategy):
        self.strategy = strategy

    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        try:
            return self.strategy.encode(df, cols, categories)
        except CustomException as e:
            logger.error(f"Error in encoding: {e}")
            return df

class FeatureEncodingMethods:
    def __init__(self):
        pass

    methods = {
        "onehot":OneHotEncoding(),
        "label":LabelEncoding(),
        "ordinal":OrdinalEncoding(),
       
    }

    @track_performance
    @staticmethod
    def feature_encoded_df(df:pd.DataFrame,cols:list,chosen_strategy:str,categories=None) -> pd.DataFrame:
        try:
            if chosen_strategy in FeatureEncodingMethods.methods:
                data_encoded = DataFrameEncoder(strategy=FeatureEncodingMethods.methods[chosen_strategy])
                df_encoded = data_encoded.encode(df,cols, categories=categories)
                return df_encoded        
            else:
                logger.error("Invalid feature encoding strategy chosen.")

        except CustomException as ce:
            logger.error(f"Exception found: {ce}")
    