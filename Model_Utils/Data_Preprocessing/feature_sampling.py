import pandas as pd
from abc import ABC, abstractmethod
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from Common_Utils import logger, CustomException, track_performance

# Strategy Interface
class OversamplingStrategy(ABC):
    @abstractmethod
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series, feature_df: pd.DataFrame, target_column:str):
        pass

# Concrete Strategies
class RandomOversampling(OversamplingStrategy):
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series,feature_df: pd.DataFrame, target_column:str):
        logger.info("Applying Random Oversampling...")
        try:
            sampler = RandomOverSampler(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_df.columns)
            y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=[target_column])
            df_balanced = pd.concat([X_train_resampled_df, y_train_resampled_df], axis=1)
            logger.info("Random Oversampling applied successfully.")
            return X_train_resampled, y_train_resampled, df_balanced 
        
        except CustomException as e:
            logger.error(f"Error in Random Oversampling: {e}")


class SMOTESampling(OversamplingStrategy):
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series, feature_df: pd.DataFrame, target_column:str):
        logger.info("Applying SMOTE Oversampling...")
        try:
            sampler = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_df.columns)
            y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=[target_column])
            df_balanced = pd.concat([X_train_resampled_df, y_train_resampled_df], axis=1)
            logger.info("SMOTE applied successfully.")
            return X_train_resampled, y_train_resampled, df_balanced 
        
        except CustomException as e:
            logger.error(f"Error in SMOTE Oversampling: {e}")
    

class ADASYNSampling(OversamplingStrategy):
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series, feature_df: pd.DataFrame, target_column:str):
        logger.info("Applying ADASYN Oversampling...")
        try:
            sampler = ADASYN(random_state=42)
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            X_train_resampled_df = pd.DataFrame(X_train_resampled, columns=feature_df.columns)
            y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=[target_column])
            df_balanced = pd.concat([X_train_resampled_df, y_train_resampled_df], axis=1)
            logger.info("ADASYN applied successfully.")
            return X_train_resampled, y_train_resampled, df_balanced 
        
        except CustomException as e:
            logger.error(f"Error in ADASYN Oversampling: {e}")

# Context Class
class OverSampler:
    def __init__(self, strategy: OversamplingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OversamplingStrategy):
        self._strategy = strategy

    def apply_strategy(self, X_train: pd.DataFrame, y_train: pd.Series, feature_df: pd.DataFrame, target_column:str):
        return self._strategy.apply(X_train, y_train, feature_df, target_column)

class FeatureOverSamplingMethods:
    def __init__(self):
        pass

    methods = {
        "random": RandomOversampling(),
        "smote": SMOTESampling(),
        "adasyn": ADASYNSampling()
        }
    
    @track_performance
    @staticmethod
    def oversampled_balanced_df(X_train: pd.DataFrame, y_train: pd.Series, feature_df: pd.DataFrame, target_column:str,chosen_strategy:str):
        if chosen_strategy in FeatureOverSamplingMethods.methods:
            try:
                over_sampler = OverSampler(strategy=FeatureOverSamplingMethods.methods[chosen_strategy])
                X_train_resampled, y_train_resampled, df_balanced = over_sampler.apply_strategy(X_train, y_train, feature_df, target_column)
                return X_train_resampled, y_train_resampled, df_balanced
            
            except CustomException as ce:
                logger.error(f"Exception found: {ce}")
