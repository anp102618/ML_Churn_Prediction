import os
import sys
import logging
import pandas as pd
from abc import ABC, abstractmethod
from Common_Utils import logger, CustomException, track_performance


# File size limit in MB
MAX_FILE_SIZE_MB = 100

# Abstract Strategy
class FileReaderStrategy(ABC):
    @abstractmethod
    def read(self, file_path: str) -> pd.DataFrame:
        pass

# Concrete Strategies
class CSVReader(FileReaderStrategy):
    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, index_col=[0])
        except Exception as e:
            raise CustomException(e, sys) 

class JSONReader(FileReaderStrategy):
    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_json(file_path)
        except Exception as e:
            raise CustomException(e, sys) 

class ExcelReader(FileReaderStrategy):
    def read(self, file_path: str) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            raise CustomException(e, sys) 

# Context
class FileReaderContext:
    def __init__(self, strategy: FileReaderStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: FileReaderStrategy):
        self._strategy = strategy

    def read_file(self, file_path: str) -> pd.DataFrame:

        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logging.error(f"File size exceeds limit: {file_size_mb:.2f} MB")
            raise ValueError(f"File too large: {file_size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB")

        try:
            if os.path.getsize(file_path) > 1:
                logging.info(f"Reading file: {file_path}")
                df = self._strategy.read(file_path)
                logging.info(f"Successfully read file with shape: {df.shape} and size: {df.size}")
            return df
        except Exception as e:
            raise CustomException(e, sys)


class FileReader:
    def __init__(self):
        pass

    ext = {
        ".csv": CSVReader(),
        ".json": JSONReader(),
        ".xls": ExcelReader(),
        ".xlsx": ExcelReader(), 
        }
    
    @track_performance
    @staticmethod
    def read_file (file_path: str) -> pd.DataFrame:
        try:
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if  file_ext in FileReader.ext:
                context = FileReaderContext(strategy=FileReader.ext[file_ext])
                df = context.read_file(file_path)
                return df

        except CustomException as ce:
            logger.error(f"Exception found: {ce}")