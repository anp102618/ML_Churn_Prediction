from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow.pyfunc
import joblib
from mlflow.tracking import MlflowClient
import yaml
from Model_Utils.Data_Preprocessing.feature_encoding import FeatureEncodingMethods
app = FastAPI(title="MLflow Production Model API")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def load_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

const = load_yaml("./constants.yaml")
model_name = const["model_name"]

# Load latest production model
def load_production_model():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_version = [
        v for v in versions if v.tags.get("version_status") == "production"
    ]
    if not prod_version:
        raise Exception("No production model found.")
    
    run_id = prod_version[0].run_id
    model_uri = f"runs:/{run_id}/{model_name}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

model = load_production_model()

ordinal_encoder = joblib.load("F:\ML_project\Script\ordinal_encoder.joblib")
ohe_encoder = joblib.load("F:\ML_project\Script\ohe_encoder.joblib")
scaler = joblib.load("F:\ML_project\Script\scaler.joblib")

# Define input schema using Pydantic
class InputData(BaseModel):
    Customer_Age: float
    Gender: str
    Dependent_count: float
    Education_Level: str
    Marital_Status: str
    Income_Category: str
    Card_Category: str
    Months_on_book: float
    Total_Relationship_Count: float
    Months_Inactive_12_mon: float
    Contacts_Count_12_mon: float
    Credit_Limit: float
    Total_Revolving_Bal: float
    Avg_Open_To_Buy: float
    Total_Amt_Chng_Q4_Q1: float
    Total_Trans_Amt: float
    Total_Trans_Ct: float
    Total_Ct_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float

ordinal_cols = ['Card_Category', 'Education_Level', 'Income_Category']
categories = {"Education_Level":['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
        "Income_Category":['Unknown', 'Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +'],
        "Card_Category":['Blue', 'Silver', 'Gold', 'Platinum'],
        }
ohe_cols = ['Gender', 'Marital_Status']
        


@app.post("/")
def home():
    return("message: Welcome to Bank Customer Churn Prediction API")

@app.post("/predict")
def predict_churn(data: InputData):
    try:
        input_df = pd.DataFrame([data.model_dump()])

        # Apply ordinal encoding
        ordinal_cols = ['Card_Category', 'Education_Level', 'Income_Category']
        input_df[ordinal_cols] = ordinal_encoder.transform(input_df[ordinal_cols])

        # Apply one-hot encoding
        onehot_cols = ["Gender", "Marital_Status"]
        encoded = ohe_encoder.transform(input_df[onehot_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe_encoder.get_feature_names_out(onehot_cols))
        
        # Drop original and merge one-hot encoded
        input_df = input_df.drop(columns=onehot_cols).reset_index(drop=True)
        input_df = pd.concat([input_df, pd.DataFrame(encoded_df)], axis=1)

        # Apply scaler
        final_columns = input_df.columns.tolist()

        # Scale
        input_scaled = scaler.transform(input_df)

        # Convert back to DataFrame with correct feature names
        input_scaled_df = pd.DataFrame(input_scaled, columns=final_columns)


        # Prediction
        prediction = model.predict(input_scaled_df)
            
        if prediction[0] == 1:
            return {"result: Attrited Customer"}
        
        else:
            return {"result: Existing Customer"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
