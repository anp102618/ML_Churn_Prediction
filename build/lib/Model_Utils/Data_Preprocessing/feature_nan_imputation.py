import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from Common_Utils import logger, CustomException, track_performance
from abc import ABC, abstractmethod

#
# Step 1: Define an Abstract Class for Imputation Strategy
class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

# Step 2: Implement Different Imputation Strategies

## 1. Mean, Median, Mode Imputation , strategy = 'mean', 'median', 'most_frequent'
class StatisticalImputationStrategy(ImputationStrategy):
    def __init__(self, strategy:str, strategy_name:str):
        self.strategy = strategy
        self.strategy_name = strategy_name

    def impute(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            imputer = SimpleImputer(strategy=self.strategy)
            df_copy = df.copy()
            df_copy[columns] = imputer.fit_transform(df_copy[columns])
            logger.info(f"Applied {self.strategy_name} Imputation on columns: {columns}")
            return df_copy
        except CustomException as e:
            logger.error(f"Error in {self.strategy_name} Imputation: {e}")
        

## 2. KNN and Iterative (MICE) Imputation
class AdvancedImputationStrategy(ImputationStrategy):
    def __init__(self, imputer, imputer_name:str):
        self.imputer = imputer
        self.imputer_name = imputer_name

    def impute(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            df_copy[columns] = self.imputer.fit_transform(df_copy[columns])
            logger.info(f"Applied {self.imputer_name} on columns: {columns}")
            return df_copy
        except CustomException as e:
            logger.error(f"Error in {self.imputer_name}: {e}")
        

        

## 3 Predictive Model Imputation : Linear Regression, DecisionTreeRegressor, Random Forest
class PredictiveModelImputationStrategy(ImputationStrategy):
    def __init__(self, model,model_name:str):
        self.model = model
        self.model_name = model_name

    def impute(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            df_copy = df.copy()
            for col in columns:
                missing_mask = df_copy[col].isnull()
                
                if missing_mask.sum() > 0:
                    predictors = df_copy.drop(columns=[col]).select_dtypes(include=[np.number])
                    
                    if predictors.shape[1] == 0:
                        logger.warning(f"Skipping {self.model_name} for {col} (no numerical predictors).")
                        continue

                    imputer = SimpleImputer(strategy="mean")
                    predictors_imputed = pd.DataFrame(imputer.fit_transform(predictors), columns=predictors.columns)
                    
                    train_data = predictors_imputed.loc[~missing_mask, predictors.columns]
                    train_target = df_copy.loc[~missing_mask, col]
                    test_data = predictors_imputed.loc[missing_mask, predictors.columns]
                    
                    model = self.model
                    model.fit(train_data, train_target)
                    
                    df_copy.loc[missing_mask, col] = model.predict(test_data)
                    logger.info(f"Applied {self.model_name} imputation on column: {col}")
                    
            return df_copy
        except CustomException as e:
            logger.error(f"Error in {self.model_name} Imputation: {e}")





# Step 3: Context Class to Apply Different Imputation Strategies
class DataImputer:
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ImputationStrategy):
        self._strategy = strategy

    def apply_imputation(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            if not all(col in df.columns for col in columns):
                raise ValueError(f"Some columns {columns} are missing from the DataFrame.")
            return self.strategy.impute(df, columns)
        except CustomException as e:
            logger.error(f"Imputation failed: {e}")
         

# Usage Example
class FeatureImputationMethods:
    def __init__(self):
        pass

    methods = {
        "mean": StatisticalImputationStrategy('mean', "mean"),
        "median": StatisticalImputationStrategy('median', "median"),
        "mode": StatisticalImputationStrategy('most_frequent', "mode"),
        "knn": AdvancedImputationStrategy(KNNImputer(n_neighbors=3), "knn_imputer"),
        "iterative": AdvancedImputationStrategy(IterativeImputer(), "iterative_imputer"),
        "linear_regression": PredictiveModelImputationStrategy(LinearRegression(),"Linear_Regression"),
        "decision_tree_regression": PredictiveModelImputationStrategy(DecisionTreeRegressor(random_state=42),"Decision_Tree_Regression"),
        "random_forest_regression": PredictiveModelImputationStrategy(RandomForestRegressor(n_estimators=100, random_state=42),"Random_Forest_Regression"),
    }

    @track_performance
    @staticmethod
    def missing_value_imputed_df(df:pd.DataFrame, cols:list,chosen_strategy:str) -> pd.DataFrame:
        try:

            if chosen_strategy in FeatureImputationMethods.methods:
                data_imputer = DataImputer(strategy=FeatureImputationMethods.methods[chosen_strategy])
                df_imputed = data_imputer.apply_imputation(df, cols) 
                return df_imputed
        
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")