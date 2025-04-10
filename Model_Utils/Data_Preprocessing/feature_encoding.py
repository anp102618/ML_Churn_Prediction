import pandas as pd
import numpy as np
import joblib
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
            encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown="ignore")
            encoded_data = encoder.fit_transform(df[cols])

            # Generate column names
            encoded_cols = encoder.get_feature_names_out(cols)
            encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols)

            # Drop original categorical columns and concatenate encoded columns
            df = df.drop(columns=cols).reset_index(drop=True)
            df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
            joblib.dump(encoder, "./FastAPI/preprocessors/ohe_encoder.joblib")

            logger.info(f"One-Hot Encoding applied successfully on columns: {cols}.")
            return df

        except Exception as e:
            logger.error(f"Error in One-Hot Encoding: {e}")


class OrdinalEncoding(EncodingStrategy):
    def encode(self, df: pd.DataFrame, cols: list, categories: dict = None) -> pd.DataFrame:
        try:
            # Extract category lists in correct order
            category_list = [categories[col] for col in cols]

            # Fit encoder on multiple columns
            encoder = OrdinalEncoder(categories=category_list)
            df[cols] = encoder.fit_transform(df[cols])

            joblib.dump(encoder, "./FastAPI/preprocessors/ordinal_encoder.joblib")
            logger.info(f"Ordinal Encoding applied successfully on columns: {cols}.")
            return df

        except Exception as e:
            logger.error(f"Error in Ordinal Encoding: {e}")
            raise CustomException(f"Ordinal Encoding failed: {e}")

            


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
    