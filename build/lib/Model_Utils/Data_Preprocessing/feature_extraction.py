from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
from Common_Utils import logger, CustomException, track_performance

class FeatureExtractionStrategy(ABC):
    @abstractmethod
    def extract_features(self, X_train, X_test, y_train=None):
        pass

class PCAFeatureExtractionStrategy(FeatureExtractionStrategy):
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, random_state=42)

    def extract_features(self, X_train, X_test, y_train=None):
        try:
            logger.info(f" Applying PCA with {self.n_components} components...")
            X_train_pca = self.pca.fit_transform(X_train)
            X_test_pca = self.pca.transform(X_test)
            explained_variance = sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA applied: Retained Variance = {explained_variance:.4f}")
            feature_names = [f"PCA_{i+1}" for i in range(self.n_components)]
            X_train_pca_df = pd.DataFrame(X_train_pca, columns=feature_names, index=X_train.index)
            X_test_pca_df = pd.DataFrame(X_test_pca, columns=feature_names, index=X_test.index)
            logger.info(f"PCA applied successfully for feature exraction with {self.n_components} selected features.")
            return X_train_pca,  X_test_pca, X_train_pca_df, X_test_pca_df
        
        except CustomException as e:
            logger.error(f"PCA extraction failed: {e}")

    
class LDAFeatureExtractionStrategy(FeatureExtractionStrategy):
    def __init__(self, n_components):
        self.n_components = n_components
        self.lda = LDA(n_components=self.n_components)

    def extract_features(self, X_train, X_test, y_train=None):
        try:
            if y_train is None:
                raise ValueError("LDA requires class labels (y_train).")

            logger.info(f"🔹 Applying LDA with {self.n_components} components...")
            X_train_lda = self.lda.fit_transform(X_train, y_train)
            X_test_lda = self.lda.transform(X_test)
            feature_names = [f"LDA_{i+1}" for i in range(self.n_components)]
            X_train_lda_df = pd.DataFrame(X_train_lda, columns=feature_names, index=X_train.index)
            X_test_lda_df = pd.DataFrame(X_test_lda, columns=feature_names, index=X_test.index)
            logger.info("LDA applied successfully for feature exraction with {self.n_components} selected features.")
            return X_train_lda, X_test_lda, X_train_lda_df, X_test_lda_df
        
        except CustomException as e:
            logger.error(f"LDA extraction failed: {e}")


class SVDFeatureExtractionStrategy(FeatureExtractionStrategy):
    def __init__(self, n_components):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)

    def extract_features(self, X_train, X_test, y_train=None):
        try:
            logger.info(f"Applying SVD with {self.n_components} components...")
            X_train_svd = self.svd.fit_transform(X_train)
            X_test_svd = self.svd.transform(X_test)
            explained_variance = sum(self.svd.explained_variance_ratio_)
            logger.info(f" SVD applied: Retained Variance = {explained_variance:.4f}")
            feature_names = [f"SVD_{i+1}" for i in range(self.n_components)]
            X_train_svd_df = pd.DataFrame(X_train_svd, columns=feature_names, index=X_train.index)
            X_test_svd_df = pd.DataFrame(X_test_svd, columns=feature_names, index=X_test.index)
            logger.info("SVD applied successfully for feature exraction with {self.n_components} selected features.")
            return X_train_svd, X_test_svd, X_train_svd_df, X_test_svd_df
        
        except CustomException as e:
            logger.error(f"SVD extraction failed: {e}")


        
class FeatureExtractor:
    def __init__(self, strategy: FeatureExtractionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: FeatureExtractionStrategy):
        self._strategy = strategy

    def extract(self, X_train, X_test, y_train=None):
        return self.strategy.extract_features(X_train, X_test, y_train)


class FeatureExtractionMethods:
    def __init__(self):
        pass
# Applying Different Feature Extraction Methods
    methods = {
        "pca": PCAFeatureExtractionStrategy(n_components=10),
        "lda": LDAFeatureExtractionStrategy(n_components=1),
        "svd": SVDFeatureExtractionStrategy(n_components=2),
        }
    
    @track_performance
    @staticmethod
    def extracted_features_df(X_train, X_test, y_train, chosen_strategy:str):
        try:

            if chosen_strategy in FeatureExtractionMethods.methods:
                feature_extractor = FeatureExtractor(strategy=FeatureExtractionMethods.methods[chosen_strategy])
                X_train_reduced, X_test_reduced, df_train_modified, df_test_modified = feature_extractor.extract(X_train, X_test, y_train=None)
                logger.info(f"\n{chosen_strategy} transformed feature shape: {X_train_reduced.shape}")
                logger.info(f"{chosen_strategy} modified train dataframe:\n{df_train_modified.head()}")
                logger.info(f"{chosen_strategy} modified test dataframe:\n{df_test_modified.head()}")
                return X_train_reduced, X_test_reduced, df_train_modified, df_test_modified
            
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")