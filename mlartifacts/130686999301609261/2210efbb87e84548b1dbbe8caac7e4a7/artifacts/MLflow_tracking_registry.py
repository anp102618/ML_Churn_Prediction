import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, confusion_matrix, classification_report
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from mlflow.models import infer_signature
from Common_Utils import logger, CustomException, load_yaml,load_joblib,track_performance,params_dict, append_constants
from Model_Utils.MLflow_Process.model_register_tag import register_and_tag_model, logging_metrics, logging_confusion_matrix, logging_params


@track_performance
def main():
    try:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_model = "Churn_Prediction" + str(timestamp)

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(current_model)

        logger.info(f"Commencing Model Tracking and Registry in MLflow..")

        const_path = "./constants.yaml"
        config_path = "./config_path.yaml"

        dict_file = load_yaml(config_path)
        const_file = load_yaml(const_path)
        X_test_data_path = dict_file["MLflowTrackingRegistry"]["X_test_data_path"]
        y_test_data_path = dict_file["MLflowTrackingRegistry"]["y_test_data_path"]
        model_path = "./"+ str(const_file["final_model_path"])

        logger.info(f"model path is : {model_path}")


        X_test = pd.read_csv(X_test_data_path)
        y_test = pd.read_csv(y_test_data_path).squeeze("columns")
        model = load_joblib(model_path)
        model_name = model.__class__.__name__

        logger.info(f"joblib model loaded successfully .. with class name :{model_name}")

        with mlflow.start_run() as run:
            logger.info(f"prediction begins ")
            y_pred = model.predict(X_test)
            logger.info(f" model prediction done")

            #Log model metrics artifact
            logging_metrics(const_path=const_path, y_test=y_test, y_pred=y_pred)

            #Log model confusion matrix
            logging_confusion_matrix(y_test=y_test, y_pred=y_pred, model_name=model_name)

            # Log model parameters
            logging_params(model_path=model_path, const_path=const_path, model_name= model_name)
            

            # Log artifacts
            mlflow.log_artifact(model_path)
            logger.info(f"Successfully logged model path ..")
            
            # Log the source code file
            mlflow.log_artifact(__file__)
            logger.info(f"Successfully logged source code file..")

            signature = infer_signature(X_test,model.predict(X_test))
            run_id_value = run.info.run_id

            append_constants(yaml_path=const_path, key="run_id", value= run_id_value)
            logger.info(f"model run_id succesfully appended in: {const_path}")

            mlflow.sklearn.log_model(model,model_name,signature=signature)
            logger.info(f"Successfully logged final model {model_name}..")

            append_constants(yaml_path=const_path, key="model_name", value= model_name)
            logger.info(f"model name succesfully appended in: {const_path}")

            
            register_and_tag_model(const_path= const_path, run_id_value= run_id_value, model_name= model_name, stage= "Staging")

    except CustomException as ce:
        logger.error(f"Exception found: {ce}")

if __name__ == "__main__":
    main()



        