import pandas as pd 
import numpy as np
from Common_Utils import logger,CustomException,track_performance
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from Common_Utils import load_yaml,extract_zip,transfer_file
from Common_Utils.file_operations import FileReader

@track_performance
def main():
    try:
        logger.info(f"Commencing Data Ingestion..")

        config_path = "./config_path.yaml"

        dict_file = load_yaml(config_path)
        zip_path = dict_file["DataIngestion"]["zip_path"]
        collected_data_path = dict_file["DataIngestion"]["collected_data_path"]
        extracted_data_path = dict_file["DataIngestion"]["extracted_data_path"]
        data_artifacts_path = dict_file["DataIngestion"]["data_artifacts_path"]


        extract_zip(zip_path= zip_path, extract_to= collected_data_path)
        transfer_file(source_path= extracted_data_path, destination_path= data_artifacts_path )

        logger.info(f"Data Ingestion succesfully complete..")

    except CustomException as ce:
        logger.error(f"Exception found: {ce}")

if __name__ == "__main__":
    main()