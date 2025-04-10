import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    ".github/workflows/.gitkeep",

    f"Data/__init__.py",
    
    f"Data/collected_data/__init__.py",
    f"Data/data_artifacts/__init__.py",

    f"Notebooks/__init__.py",
    f"Notebooks/EDA.ipynb",
    f"Notebooks/notebook_Data_Ingestion.ipynb",
    f"Notebooks/notebook_Data_Validation.ipynb",
    f"Notebooks/notebook_Data_Transformation.ipynb",
    f"Notebooks/notebook_Model_Training_Evaluation.ipynb",
    f"Notebooks/notebook_Model_Implementation.ipynb",

    f"src/__init__.py",
    f"src/Data_Ingestion.py",
    f"src/Data_Validation.py",
    f"src/Data_Transformation.py",
    f"src/Model_Training_Evaluation.py",
    f"src/Model_Implementation.py",

    f"Model_Utils/__init__.py",
    f"Model_Utils/Data_Preprocessing/__init__.py",
    f"Model_Utils/Data_Preprocessing/feature_nan_imputation.py",
    f"Model_Utils/Data_Preprocessing/feature_outlier_detection.py",
    f"Model_Utils/Data_Preprocessing/feature_outlier_handling.py",
    f"Model_Utils/Data_Preprocessing/feature_sampling.py",
    f"Model_Utils/Data_Preprocessing/feature_encoding.py",
    f"Model_Utils/Data_Preprocessing/feature_selection.py",
    f"Model_Utils/Data_Preprocessing/feature_extraction.py",
    f"Model_Utils/Data_Preprocessing/feature_scaling.py",
    f"Model_Utils/Model_Selection/__init__.py",
    f"Model_Utils/Model_Selection/models_initialization.py",
    f"Model_Utils/Model_Selection/Models_training_evaluation.py",
    f"Model_Utils/Model_Selection/supervised_tuned_models.py",
    f"Model_Utils/Model_Selection/unsupervised_tuned_models.py",
    f"Model_Utils/MLflow_Process/__init__.py",
    f"Model_Utils/MLflow_Process/model_register_tag.py",

    f"Yaml_Repo/__init__.py",
    f"Yaml_Repo/classifiers_param.yaml",
    f"Yaml_Repo/regressors_param.yaml",
    f"Yaml_Repo/unsupervised_param.yaml",
    f"Yaml_Repo/schema_data.yaml",
    f"Yaml_Repo/tuned_params_metrics.yaml",

    f"Script/__init__.py",
    f"Script/test.py",
    f"Script/production.py",
    
    f"Final_Model/__init__.py",

    f"Common_Utils/__init__.py",
    f"Common_Utils/dataframe_methods.py",
    f"Common_Utils/file_operations.py",

    f"FastAPI/__init__.py",
    f"FastAPI/app.py",
    f"FastAPI/preprocessors/__init__.py",
    f"FastAPI/requirements.txt",
    f"FastAPI/input.json",



    "config_path.yaml",
    "constants.yaml",
    "requirements.txt",
    "setup.py",
    "misc.py",
 


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")