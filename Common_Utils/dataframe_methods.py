import pandas as pd
import numpy as np
import os
from Common_Utils import logger, CustomException, track_performance
from sklearn.model_selection import train_test_split


class DataFrameMethods:
    def __init__(self):
        pass

    def numerical_column_list(df):
        try:
            columns_numerical = df.select_dtypes(include=["int32","int64","float32","float64"]).columns.tolist()
            logger.info(f'"numerical columns: Total:{len(columns_numerical)} with columns_list: {sorted(columns_numerical)}"')
            return columns_numerical
        
        except CustomException as ce:
            logger.error(f"Exception found in numerical_column_list: {ce}")
    
    def categorical_column_list(df):
        try:
            columns_categorical = df.select_dtypes(include=["object","category"]).columns.tolist()
            logger.info(f'"categorical columns: Total:{len(columns_categorical)} with columns_list: {sorted(columns_categorical)}"')
            return columns_categorical
        
        except CustomException as ce:
            logger.error(f"Exception found in categorical_column_list: {ce}")
        
    def missing_values_columns(df):
        try:
            missing_percentage = (df.isnull().sum() / len(df)) * 100
            missing_columns = missing_percentage[missing_percentage > 0]
            summary_df = pd.DataFrame({
                'Column Name': missing_columns.index,
                'Missing Value Percentage': missing_columns.values,
                'Data Type': df[missing_columns.index].dtypes.values
            }).reset_index(drop=True)

            summary_df = summary_df.sort_values(by='Missing Value Percentage', ascending=False)
            logger.info(f"missing_values_columns computed")
            return summary_df
        
        except CustomException as ce:
            logger.error(f"Exception found in missing_values_columns : {ce}")

    def unique_values(df,dtype:list):
        try:
            a,b,c,d=[],[],[],[]
            for col in df.columns:
                if df[col].dtype in dtype:
                    a.append(col)
                    b.append(df[col].nunique())
                    c.append(df[col].unique().tolist())
                    d.append(df[col].value_counts().idxmax())
                    
            df = pd.DataFrame({"column_name":a, "no.":b, "values":c, "mode":d})
            logger.info(f"unique_values per column computed")
            return df.sort_values(by = "no.", ascending = False)
        
        except CustomException as ce:
            logger.error(f"Exception found in unique_values : {ce}")
    
    def column_category_percentage(df,column):
        try:
    
            if column in df.columns and len(df[column])>1:
                df = df.copy()
                value_counts = df[column].value_counts(dropna=False)
                total_count = len(df[column])
                summary_df = pd.DataFrame({"category":value_counts.index, "Count":value_counts.values,"%":(value_counts.values/total_count)*100})
                logger.info(f"column_category_percentage computed")
                return summary_df.sort_values(by = "Count", ascending = False)
        
        except CustomException as ce:
            logger.error(f"Exception found in column_category_percentage : {ce}")




    def descriptive_statistics(df,target:str):
        try:
            stats = {}
            df = df.dropna()
            for column in df.columns:

                if df[column].dtype in ["int32","int64","float32","float64"]:
                    col_data = df[column]
                    stats[column] = {
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'variance': col_data.var(),
                        'std_dev': col_data.std(),
                        'skewness': col_data.skew(),
                        'kurtosis': col_data.kurtosis(),
                        'min': col_data.min(),
                        '25%': col_data.quantile(0.25),
                        '50%': col_data.median(),
                        '75%': col_data.quantile(0.75),
                        'max': col_data.max(),
                        'deviation%_mean': ((col_data > col_data.mean()).mean() - (col_data < col_data.mean()).mean())* 100,
                        #'correlation': df[column].corr(df['target'], method='spearman')
                    }

            stats_df = pd.DataFrame(stats)
            logger.info(f"Descriptive Statistics computed")
            return stats_df
        
        except CustomException as ce:
            logger.error(f"Exception found in descriptive_statistics : {ce}")

    def drop_columns(df, cols):
        if len(df) > 1:
            df_modified = df.drop(cols , axis=1)
            if all(col not in df.columns for col in cols):
                logger.info(f'"columns {cols} dropped from dataframe')
            return df_modified
        else:
            print(f'"Dataframe is empty"')  

    def categorical_columns_information(df:pd.DataFrame):
        num_col= DataFrameMethods.numerical_column_list(df)
        cat_col = DataFrameMethods.categorical_column_list(df)
        logger.info(f'"Missing Values Columns:{os.linesep}{ DataFrameMethods.missing_values_columns(df)}"')
        logger.info(f'"Categorical Columns Detail:{os.linesep}{DataFrameMethods.unique_values(df,["object","category"])}"')
        

    def split_train_test(df, target_column, test_size=0.2, random_state=42):
        
        try:
            # Validate inputs
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Input data must be a pandas DataFrame.")
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

            if not (0 < test_size < 1):
                raise ValueError("Test size must be between 0 and 1.")

            logger.info("Splitting dataset into training and testing sets...")
            
            # Splitting dataset
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            logger.info(f"Train set size: {X_train.shape[0]} samples")
            logger.info(f"Test set size: {X_test.shape[0]} samples")
            
            return X_train, X_test, y_train, y_test , X , y
        
        except CustomException as e:
            logger.error(f"Error during train-test split: {e}")
            return None, None, None, None

     

####################################################################

class DateTimeExtractor:
    def __init__(self, df, column_name):
       
        self.df = df.copy()
        self.column_name = column_name
        self.validate_column()
        self.convert_to_datetime()

    def validate_column(self):
        if self.column_name not in self.df.columns:
            raise ValueError(f"Column '{self.column_name}' does not exist in the DataFrame.")

    def convert_to_datetime(self):
        self.df[self.column_name] = pd.to_datetime(self.df[self.column_name], errors='coerce')
        if self.df[self.column_name].isnull().all():
            raise ValueError(f"All values in column '{self.column_name}' could not be converted to datetime.")

    def extract_date(self):
        self.df[f'{self.column_name}_date'] = self.df[self.column_name].dt.date
        return self.df

    def extract_time(self):
        self.df[f'{self.column_name}_time'] = self.df[self.column_name].dt.time
        return self.df

    def extract_day(self):
        self.df[f'{self.column_name}_day'] = self.df[self.column_name].dt.day
        return self.df

    def extract_month(self):
        self.df[f'{self.column_name}_month'] = self.df[self.column_name].dt.month
        return self.df
    
    def extract_month_name(self, locale='en_US.utf8'):
        self.df[f'{self.column_name}_month_name'] = self.df[self.column_name].dt.month_name(locale=locale)
        return self.df

    def extract_quarter(self):
        self.df[f'{self.column_name}_quarter'] = self.df[self.column_name].dt.quarter
        return self.df

    def extract_year(self):
        self.df[f'{self.column_name}_year'] = self.df[self.column_name].dt.year
        return self.df

    def extract_all(self,locale='en_US.utf8'):
        self.extract_date()
        self.extract_time()
        self.extract_day()
        self.extract_month()
        self.extract_month_name(locale=locale)
        self.extract_quarter()
        self.extract_year()
        return self.df


