import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import shap
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from abc import ABC, abstractmethod
from Common_Utils import logger, CustomException, track_performance



# Abstract Strategy Class
class FeatureSelectionStrategy(ABC):
    @abstractmethod
    def select_features(self, X_train, X_test, y_train):
        pass

# Variance Threshold Method
class VarianceThresholdSelector(FeatureSelectionStrategy):
    def __init__(self, threshold=0.01):
        self.threshold = threshold
    
    def select_features(self, X_train, X_test, y_train=None):
        try:
            selector = VarianceThreshold(threshold=self.threshold)
            X_train_selected = selector.fit_transform(X_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"VarianceThreshold applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df , selected_features
        except CustomException as e:
            logger.error(f"Error in VarianceThresholdSelector: {e}")
            

#  Correlation-Based Selection (Pearson or Spearman)
class CorrelationSelector(FeatureSelectionStrategy):
    def __init__(self, method="pearson", threshold=0.5):
        self.method = method
        self.threshold = threshold

    def select_features(self, X_train, X_test, y_train):
        try:
            selected_features = []
            for i in range(X_train.shape[1]):
                feature = X_train.iloc[:, i]
                if self.method == "pearson":
                    corr, _ = pearsonr(feature, y_train)
                else:
                    corr, _ = spearmanr(feature, y_train)
                if abs(corr) >= self.threshold:
                    selected_features.append(i)
            
            X_train_selected = X_train.iloc[:, selected_features]
            X_test_selected = X_test.iloc[:, selected_features]
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"Correlation ({self.method}) applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df, selected_features
        except CustomException as e:
            logger.error(f"Error in CorrelationSelector: {e}")
            

# Chi-Square Test
class ChiSquareSelector(FeatureSelectionStrategy):
    def __init__(self, k):
        self.k = k

    def select_features(self, X_train, X_test, y_train):
        try:
        # Apply Chi-Square test (requires non-negative values)
                scaler = MinMaxScaler()  
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                chi_selector = SelectKBest(chi2, k=self.k)
                X_train_selected = chi_selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = chi_selector.transform(X_test_scaled)
                selected_features = X_train.columns[chi_selector.get_support()].tolist()
                X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
                X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

                logger.info(f"Chi-Square applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
                return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df, selected_features

        except CustomException as e:
            logger.error(f"Error in ChiSquareSelector: {e}")
            

# ANOVA F-Test
class ANOVASelector(FeatureSelectionStrategy):
    def __init__(self, k=5):
        self.k = k

    def select_features(self, X_train, X_test, y_train):
        try:
            selector = SelectKBest(score_func=f_classif, k=min(self.k, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()].tolist()
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"ANOVA F-Test applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df, selected_features
        except CustomException as e:
            logger.error(f"Error in ANOVASelector: {e}")
            


# Information Gain (Entropy-Based Selection)
class InformationGainSelector(FeatureSelectionStrategy):
    def __init__(self, k=5):
        self.k = k

    def select_features(self, X_train, X_test, y_train):
        try:
            discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
            X_train_discretized = discretizer.fit_transform(X_train)
            X_test_discretized =discretizer.transform(X_test)
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.k, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train_discretized, y_train)
            X_test_selected = selector.transform(X_test_discretized)
            selected_features = X_train.columns[selector.get_support()]
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"Information Gain applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df, selected_features

        except CustomException as e:
            logger.error(f"Error in InformationGainSelector: {e}")



# SHAP-Based Feature Selection        
class SHAPSelector(FeatureSelectionStrategy):
    def __init__(self, model=None, n_features=5):
        self.model = model if model else XGBClassifier()
        self.n_features = n_features

    def select_features(self, X_train, X_test, y_train):
        try:
            self.model.fit(X_train, y_train)
            explainer = shap.Explainer(self.model, X_train)
            shap_values = explainer(X_train)
            importance = np.abs(shap_values.values).mean(axis=0)
            top_indices = np.argsort(importance)[-self.n_features_features:]
            X_train_selected = X_train.iloc[:, top_indices]
            X_test_selected = X_test.iloc[:, top_indices]
            selected_features = (X_train.columns[top_indices]).tolist()
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"SHAP Feature Selection applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected, X_train_selected_df, X_test_selected_df, selected_features

        except CustomException as e:
            logger.error(f"Error in SHAPSelector: {e}")
            


# Elastic Net (Combination of L1 & L2)
class ElasticNetSelector(FeatureSelectionStrategy):
    def __init__(self, alpha=1.0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def select_features(self, X_train, X_test, y_train):
        try:
            elastic_net = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
            elastic_net.fit(X_train, y_train)
            selector = SelectFromModel(elastic_net, prefit=True)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            selected_features = X_train.columns[selector.get_support()]
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"Elastic Net Selection applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected,X_train_selected_df, X_test_selected_df,selected_features.tolist()

        except CustomException as e:
            logger.error(f"Error in ElasticNetSelector: {e}")
           

# Tree-Based Feature Importance (Random Forest)
class TreeBasedSelector(FeatureSelectionStrategy):
    def __init__(self, model_type="random_forest", n_estimators=100):
        self.model_type = model_type
        self.n_estimators = n_estimators

    def select_features(self, X_train, X_test, y_train):
        try:
            if self.model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
            else:
                model = GradientBoostingClassifier(n_estimators=self.n_estimators, random_state=42)
            
            model.fit(X_train, y_train)
            feature_importances = model.feature_importances_
            selected_indices = feature_importances > np.percentile(feature_importances, 25)  # Keep top 75% important features
            
            X_train_selected = X_train.iloc[:, selected_indices]
            X_test_selected = X_test.iloc[:, selected_indices]
            selected_features = X_train_selected.columns.tolist()
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
            logger.info(f"Tree based Selection applied. Features reduced from {X_train.shape[1]} to {X_train_selected.shape[1]}")
            return X_train_selected, X_test_selected,X_train_selected_df, X_test_selected_df,selected_features
        
        except CustomException as e:
            logger.error(f"Error in ElasticNetSelector: {e}")



# Context Class: Feature Selector
class FeatureSelector:
    def __init__(self, strategy: FeatureSelectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureSelectionStrategy):
        self._strategy = strategy
    
    def select(self, X_train, X_test, y_train):
        return self.strategy.select_features(X_train, X_test, y_train)


class FeatureSelectionMethods:
    def __init__(self):
        pass

    methods = {
        
        "variance_threshold": VarianceThresholdSelector(threshold=0.1),
        "pearson_correlation": CorrelationSelector(method="pearson", threshold=0.3),
        "spearman_correlation": CorrelationSelector(method="spearman", threshold=0.3),
        "chi_square": ChiSquareSelector(k=2),
        "anova_ftest": ANOVASelector(k=2),
        "information_gain": InformationGainSelector(k=2),
        "shap": SHAPSelector(model=XGBClassifier(), n_features=5),
        "elasticnet": ElasticNetSelector(alpha=0.01, l1_ratio=0.5),
        "random_forest": TreeBasedSelector(model_type="random_forest", n_estimators=100),
        "gradient_boosting": TreeBasedSelector(model_type="gradient_boosting", n_estimators=100),
         }
    
    @track_performance
    @staticmethod
    def selected_features_df(X_train, X_test, y_train,chosen_strategy:str):
        try:
            if chosen_strategy in FeatureSelectionMethods.methods:
                feature_selector = FeatureSelector(strategy=FeatureSelectionMethods.methods[chosen_strategy])
                X_train_selected, X_test_selected,X_train_selected_df, X_test_selected_df,selected_features = feature_selector.select(X_train, X_test, y_train)
                logger.info(f"\n{chosen_strategy} selected features: {selected_features}")
                return X_train_selected, X_test_selected,X_train_selected_df, X_test_selected_df,selected_features
        
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")
