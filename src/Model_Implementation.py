import pandas as pd 
import numpy as np
from Common_Utils import logger,CustomException,track_performance
import os
import sys
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from Common_Utils import load_yaml,extract_zip,transfer_file,update_final_model_path
from Common_Utils.file_operations import FileReader
from Model_Utils.Model_Selection.models_initialization import SelectedModelInitializer

@track_performance
def main():
    try:
        logger.info(f"Commencing Model_Implementation..")

        const_path = "./constants.yaml"
        config_path = "./config_path.yaml"

        dict_file = load_yaml(config_path)
        X_train_data_path = dict_file["ModelImplementation"]["X_train_data_path"]
        X_test_data_path = dict_file["ModelImplementation"]["X_test_data_path"]
        y_train_data_path = dict_file["ModelImplementation"]["y_train_data_path"]
        y_test_data_path = dict_file["ModelImplementation"]["y_test_data_path"]
        tuned_models_yaml_path = dict_file["ModelImplementation"]["tuned_models_path"]
        model_folder_path = dict_file["ModelImplementation"]["model_folder_path"]

        X_train = pd.read_csv(X_train_data_path)
        y_train = pd.read_csv(y_train_data_path).squeeze("columns")

        models_config = load_yaml(tuned_models_yaml_path)

        first_model_name = next(iter(models_config))
        model_entry = models_config[first_model_name]
        model_str = model_entry["model"]
        model_params = model_entry["parameters"]

            # Match string with dictionary class name
        matched_class = None

        # Loop through the dictionary values to find a match by class name
        for name, model_obj in SelectedModelInitializer.classifier.items():
            if model_obj.__class__.__name__ == model_str:
                matched_class = model_obj.__class__
                break

        if matched_class is None:
            raise ValueError(f"Model class {model_str} not found in dictionary.")

        # Instantiate and train the model
        model = matched_class(**model_params)
        model.fit(X_train, y_train)

        # Save the trained model
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_folder = os.path.join(model_folder_path, timestamp)
        os.makedirs(new_folder, exist_ok=True)
        destination_path = Path(os.path.join(new_folder, "model.joblib"))

        joblib.dump(model, destination_path)
        update_final_model_path(const_path,destination_path)

        logger.info(f"Final model successfully saved at : {destination_path}")
    
    except CustomException as ce:
        logger.error(f"Exception found: {ce}")

if __name__ == "__main__":
    main()

