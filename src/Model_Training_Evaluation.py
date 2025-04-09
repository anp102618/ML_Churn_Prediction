import pandas as pd 
import numpy as np
from Common_Utils import logger,CustomException,track_performance
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from Common_Utils import load_yaml,extract_zip,transfer_file
from Common_Utils.file_operations import FileReader
from Model_Utils.Model_Selection.supervised_tuned_models import SupervisedHyperparameterSearchMethods
from Model_Utils.Model_Selection.models_initialization import SelectedModelInitializer
from Model_Utils.Model_Selection.Models_training_evaluation import BestModelEvaluation
from Model_Utils.Model_Selection.Models_training_evaluation import SelectedModelsYaml
from Common_Utils.dataframe_methods import DataFrameMethods

@track_performance
def main():
    try:
        logger.info(f"Commencing Model_Training_Evaluation..")

        config_path = "./config_path.yaml"

        dict_file = load_yaml(config_path)
        raw_data_data_path = dict_file["DataValidation"]["raw_data_path"]
        X_train_data_path = dict_file["ModelTrainingEvaluation"]["X_train_data_path"]
        X_test_data_path = dict_file["ModelTrainingEvaluation"]["X_test_data_path"]
        y_train_data_path = dict_file["ModelTrainingEvaluation"]["y_train_data_path"]
        y_test_data_path = dict_file["ModelTrainingEvaluation"]["y_test_data_path"]
        tuned_models_yaml_path = dict_file["ModelTrainingEvaluation"]["tuned_models_path"]
        classifiers_param_path = dict_file["ModelTrainingEvaluation"]["classifiers_param_path"]

        X_train = pd.read_csv(X_train_data_path)
        y_train = pd.read_csv(y_train_data_path).squeeze("columns")
        X_test = pd.read_csv(X_test_data_path)
        y_test = pd.read_csv(y_test_data_path).squeeze("columns")
        



        dict_sup_tuned_hyperparams = SupervisedHyperparameterSearchMethods.tuned_model_parameters(models=SelectedModelInitializer.classifier,
                                                                                                X_train= X_train , y_train=y_train,
                                                                                                scoring= "accuracy",
                                                                                                yaml_file=classifiers_param_path,
                                                                                                chosen_strategy="grid_search_cv")


        df_tuned_models = BestModelEvaluation.evaluate(model_dict=SelectedModelInitializer.classifier,
                                            param_dict=dict_sup_tuned_hyperparams,
                                            X_train= X_train,
                                            X_test= X_test,
                                            y_train= y_train,
                                            y_test= y_test,
                                            scoring= "accuracy_test"
                                            )
        

        SelectedModelsYaml.tuned_models_yaml(models_dict=SelectedModelInitializer.classifier,
                                     params_dict=dict_sup_tuned_hyperparams,
                                     metrics_df=df_tuned_models,
                                     yaml_path=tuned_models_yaml_path,
                                     scoring="accuracy_test")
        
        logger.info(f"Model_Training_Evaluation succesfully complete..")

    except CustomException as ce:
        logger.error(f"Exception found: {ce}")

if __name__ == "__main__":
    main()