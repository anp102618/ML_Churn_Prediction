from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from scipy import stats
from Common_Utils import logger, CustomException, track_performance


# Strategy Interface
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, data):
        pass

# Concrete Strategies
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, data):
        z_scores = np.abs(stats.zscore(data))
        return np.where(z_scores > 3, True, False)

class IQRBasedOutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, data):
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where((data < lower_bound) | (data > upper_bound), True, False), lower_bound, upper_bound

class SVMOutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, data):
        model = OneClassSVM(nu=0.05, kernel="rbf")
        model.fit(data.reshape(-1, 1))
        return model.predict(data.reshape(-1, 1)) == -1


# Context Class
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        self.strategy = strategy

    def detect(self, data):
        return self.strategy.detect_outliers(data)

class FeatureOutlierDetection:
    def __init__(self):
        pass
    
    methods = {
        "zscore": ZScoreOutlierDetection(),
        "iqr": IQRBasedOutlierDetection(),
        "svm": SVMOutlierDetection(),
    }

    @track_performance
    @staticmethod
    def detect_outliers_in_df(df:pd.DataFrame, method:str, target:str) -> pd.DataFrame:
        try:
            logger.info(f"Starting outlier detection using method: {method}")
        
            if method not in FeatureOutlierDetection.methods:
                logger.error("Invalid method selected.")
                raise ValueError("Invalid method selected.")
            
            detector = OutlierDetector(FeatureOutlierDetection.methods[method])
            
            results = []
            for column in df.columns:
                if column != target:
                    try:
                        if df[column].dtype in ['object','category']:
                            outliers_detected = df[column].value_counts(normalize=True) < 0.05
                            num_outliers = outliers_detected.sum()
                            percentage_outliers = (num_outliers / len(df)) * 100 if len(df) > 0 else 0
                            lower_bound, upper_bound = None, None

                        elif df[column].dtype in ['int32','int64','float32','float64']:
                            data = df[column].dropna().values
                            if method == "iqr":
                                outliers_detected, lower_bound, upper_bound = detector.detect(data)
                            else:
                                outliers_detected = detector.detect(data)
                                lower_bound, upper_bound = None, None
                            num_outliers = np.sum(outliers_detected)
                            percentage_outliers = (num_outliers / len(data)) * 100 if len(data) > 0 else 0
                        
                        results.append({
                            "column_name": column,
                            "number_of_outliers": num_outliers,
                            "percentage_of_outliers": percentage_outliers,
                            "column_dtype": df[column].dtype,
                        })
                        
                    
                    except CustomException as e:
                        logger.error(f"Error processing column {column}: {e}")
   
            df = pd.DataFrame(results)
            outlier_num_cols = df.loc[df['column_dtype'].apply(lambda x: x in ['int32','int64','float32','float64']), "column_name"].values
            logger.info(f"Outlier columns detected via {method}OutlierDetection: {outlier_num_cols}")
            df = df[df["percentage_of_outliers"]>0]
            sorted_df = df.sort_values(by=['percentage_of_outliers'], ascending=False)
            sorted_df.reset_index(inplace=True)
            return sorted_df, outlier_num_cols

        except CustomException as ce:
            logger.error(f"Exception found: {ce}")