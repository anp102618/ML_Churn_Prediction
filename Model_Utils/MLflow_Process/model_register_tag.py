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


def register_and_tag_model(const_path ,run_id_value, model_name, stage):
            
    try:
        # Create an MLflow client
        client = MlflowClient()

        # Create the model URI
        model_uri = f"runs:/{run_id_value}/artifacts/{model_name}"

        append_constants(yaml_path=const_path, key="model_uri", value= model_uri)
        logger.info(f"model uri succesfully appended in: {const_path}")

        # Register the model
        reg = mlflow.register_model(model_uri, model_name)

        # Get the model version
        model_version = reg.version  # Get the registered model version

        append_constants(yaml_path=const_path, key="model_version", value= model_version)
        logger.info(f"model version  succesfully appended in: {const_path}")

        # Transition the model version to Staging
        new_stage = stage

        append_constants(yaml_path=const_path, key="Current_stage", value= new_stage)
        logger.info(f"model Current stage succesfully appended in: {const_path}")

        client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="version_status",
        value=new_stage
        )

        append_constants(yaml_path=const_path, key="model_version", value= model_version)
        logger.info(f"model run_id succesfully appended in: {const_path}")

        logger.info(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")

    except CustomException as e:
            logger.error(f"Error tagging model in mlflow : {e}")


def logging_metrics(const_path, y_test, y_pred):
    try:
        model_metrics = {"accuracy":accuracy_score(y_test, y_pred),
                        "precision" : precision_score(y_test, y_pred),
                            "recall" : recall_score(y_test, y_pred),
                            "f1" : f1_score(y_test, y_pred),
                            "roc_auc": roc_auc_score(y_test, y_pred)
                        }
                
        clean_metrics = {k: float(v) if isinstance(v, np.generic) else v for k, v in model_metrics.items()}
        append_constants(yaml_path=const_path, key="metrics", value= clean_metrics)
        logger.info(f"model_metrics successfully appended in: {const_path}")

        # Log metrics
        mlflow.log_metrics(model_metrics)
        logger.info(f"Successfully logged classification metrics..")

    except CustomException as e:
            logger.error(f"Error logging model metrics in mlflow : {e}")
    

def logging_confusion_matrix(y_test, y_pred, model_name):
    
    try:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(cm_path)
        
        # Log confusion matrix artifact
        mlflow.log_artifact(cm_path)
        logger.info(f"Successfully logged confusion matrix artifact..")
     
    except CustomException as e:
        logger.error(f"Error logging model confusion matrix in mlflow : {e}")

def logging_params(model_path, const_path , model_name):
    try:
        filtered_params = params_dict(model_path)
        append_constants(yaml_path=const_path, key="params", value= filtered_params)
        logger.info(f"model_params successfully appended in: {const_path}")

        for k, v in filtered_params.items():
            mlflow.log_param(k, v)
        logger.info(f"Successfully logged {model_name} model params ..")

    except CustomException as e:
        logger.error(f"Error logging model params in mlflow : {e}")

    