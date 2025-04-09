import unittest
import yaml
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np 
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

with open("./constants.yaml", "r") as f:
            const_data = yaml.safe_load(f)

model_name= const_data["model_name"]

# Unit test class to test the loading of models frrom the "Staging" stage in MLflow
class TestModelLoading(unittest.TestCase):

#Test if the model exists in "Staging" stage 
    def test_model_in_staging(self):

        client = MlflowClient()
        all_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter versions based on custom tag (e.g., version_status == staging)
        versions = [v for v in all_versions if v.tags.get("version_status", "").lower() == "staging"]

        # Assert that atleast one version of model exists in the "Staging" stage
        self.assertGreater(len(versions), 0, "No model found in the 'Staging' stage" )

    
    def test_model_loading(self):
        
        client = MlflowClient()
        all_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter versions based on custom tag (e.g., version_status == staging)
        versions = [v for v in all_versions if v.tags.get("version_status", "").lower() == "staging"]

        if not versions:
            self.fail("No model found in 'Staging' stage , skipping model loading test.")

        latest_version  = versions[0].version
        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"
        try:
            loaded_model = mlflow.pyfunc.load_model(logged_model)

        except Exception as e :
            self.fail(f"Failer to load the model: {e}")

        self.assertIsNotNone(loaded_model, "The model is not None ")
        print(f"Model successfully loaded from {logged_model}")
        print(f"run_ id : {run_id}")

    
    def test_model_performance(self):

        client = MlflowClient()
        all_versions = client.search_model_versions(f"name='{model_name}'")

        # Filter versions based on custom tag (e.g., version_status == staging)
        versions = [v for v in all_versions if v.tags.get("version_status", "").lower() == "staging"]

        if not versions:
            self.fail("No model found in 'Staging' stage , skipping model loading test.")

        run_id = versions[0].run_id
        logged_model = f"runs:/{run_id}/{model_name}"
        loaded_model = mlflow.pyfunc.load_model(logged_model)
         
        X_test_data_path = "./Data/data_artifacts/X_test.csv"
        y_test_data_path = "./Data/data_artifacts/y_test.csv"

        if not os.path.exists(X_test_data_path):
            self.fail(f" X_test data not found at :{X_test_data_path}")
        if not os.path.exists(y_test_data_path):
            self.fail(f" X_test data not found at :{y_test_data_path}")

        X_test = pd.read_csv(X_test_data_path)
        y_test = pd.read_csv(y_test_data_path).squeeze("columns")

        y_pred = loaded_model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision  = float(precision_score(y_test, y_pred))
        recall =  float(recall_score(y_test, y_pred))
        f1  = float(f1_score(y_test, y_pred))
        roc_auc =  float(roc_auc_score(y_test, y_pred))

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"f1: {f1}")
        print(f"roc_auc: {roc_auc}")

        self.assertGreaterEqual(accuracy, 0.7, "Accuracy is below threshold ")
        self.assertGreaterEqual(precision, 0.7, "Precision is below threshold ")
        self.assertGreaterEqual(recall, 0.7, "Recall is below threshold ")
        self.assertGreaterEqual(f1, 0.7, "f1 is below threshold ")
        self.assertGreaterEqual(roc_auc, 0.7, "roc_auc is below threshold ")


if __name__ == "__main__":
    unittest.main()

