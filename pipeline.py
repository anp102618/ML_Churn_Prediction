import pandas as pd 
import numpy as np
from Common_Utils import logger,CustomException,track_performance
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from Common_Utils import load_yaml,extract_zip,transfer_file
import src.Data_Ingestion, src.Data_Transformation, src.Model_Training_Evaluation, src.Model_Implementation, src.MLflow_tracking_registry

@track_performance
def main():

    logger.info(f" ML workflow execution started ...")

    src.Data_Ingestion.main()
    src.Data_Transformation.main()
    src.Model_Training_Evaluation.main()
    src.Model_Implementation.main()
    src.MLflow_tracking_registry.main()

    logger.info(f"ML Workflow successfully completed..")

if __name__ == "__main__":
    main()