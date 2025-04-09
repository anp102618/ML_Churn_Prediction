import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import clone, is_classifier, is_regressor
from Common_Utils import logger, CustomException, track_performance



class ModelEvaluator:
    def __init__(self, models_dict, param_dict, X_train, X_test, y_train, y_test, scoring):
        self.models_dict = models_dict
        self.params_dict = param_dict
        self.scoring = scoring
        self.X_train, self.X_test, self.y_train, self.y_test =  X_train, X_test, y_train, y_test
        

    @staticmethod
    def adjusted_r2(r2, n, p):
        """Calculate Adjusted R² Score"""
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def evaluate_models(self):
        """Train and evaluate all models, returning results in a DataFrame."""
        results = []

        for short_name, model_instance in self.models_dict.items():
            model_name = type(model_instance).__name__

            if is_regressor(model_instance):

                try:
                    logger.info(f"Training {model_name}...")
                    params = self.params_dict.get(model_name, {})
                    model_instance.set_params(**params)
                    model_instance.fit(self.X_train, self.y_train)
                    y_train_pred = model_instance.predict(self.X_train)
                    y_pred = model_instance.predict(self.X_test)

                    # Calculate metrics
                    r2_test = r2_score(self.y_test, y_pred)
                    adj_r2_test = self.adjusted_r2(r2_test, len(self.y_test), self.X_test.shape[1])
                    mse_test = mean_squared_error(self.y_test, y_pred)
                    rmse_test = np.sqrt(mse_test)
                    mae_test = mean_absolute_error(self.y_test, y_pred)

                    r2_train = r2_score(self.y_train, y_train_pred)
                    adj_r2_train = self.adjusted_r2(r2_train, len(self.y_train), self.X_train.shape[1])
                    mse_train = mean_squared_error(self.y_train, y_train_pred)
                    rmse_train = np.sqrt(mse_train)
                    mae_train = mean_absolute_error(self.y_train, y_train_pred)

                    # Store results
                    results.append({
                        "Model": model_name,
                        "r2_train": r2_train,
                        "r2_test": r2_test,
                        "adj_r2_train": adj_r2_train,
                        "adj_r2_test": adj_r2_test,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                        "rmse_train": rmse_train,
                        "rmse_test": rmse_test,
                        "mae_train": mae_train,
                        "mae_test": mae_test,
                    })

                except CustomException as e:
                    logger.error(f"Error training {model_name}: {e}")

            elif is_classifier(model_instance):

                try:
                    logger.info(f"Training {model_name}...")
                    params = self.params_dict.get(model_name, {})
                    model_instance.set_params(**params)
                    model_instance.fit(self.X_train, self.y_train)
                    y_train_pred = model_instance.predict(self.X_train)
                    y_pred = model_instance.predict(self.X_test)

                    if len(np.unique(self.y_test)) > 2:
                        y_pred_prob = model_instance.predict_proba(self.X_test)
                        roc_auc_test = roc_auc_score(self.y_test_bin, y_pred_prob, average="weighted", multi_class="ovr")
                    else:
                        y_pred_prob = model_instance.predict_proba(self.X_test)[:, 1]
                        roc_auc_test = roc_auc_score(self.y_test, y_pred_prob)

                    if len(np.unique(self.y_train)) > 2:
                        y_train_pred_prob = model_instance.predict_proba(self.X_train)
                        roc_auc_train = roc_auc_score(self.y_train_bin, y_train_pred_prob, average="weighted", multi_class="ovr")
                    else:
                        y_train_pred_prob = model_instance.predict_proba(self.X_train)[:, 1]
                        roc_auc_train = roc_auc_score(self.y_train, y_train_pred_prob)

                    metrics = {
                        'Model': model_name,
                        'accuracy_train': accuracy_score(self.y_train, y_train_pred),
                        'accuracy_test': accuracy_score(self.y_test, y_pred),
                        'precision_train': precision_score(self.y_train, y_train_pred, average='weighted'),
                        'precision_test': precision_score(self.y_test, y_pred, average='weighted'),
                        'recall_train': recall_score(self.y_train, y_train_pred, average='weighted'),
                        'recall_test': recall_score(self.y_test, y_pred, average='weighted'),
                        'f1_Score_train': f1_score(self.y_train, y_train_pred, average='weighted'),
                        'f1_Score_test': f1_score(self.y_test, y_pred, average='weighted'),
                        'roc_auc_train': roc_auc_train,
                        'roc_auc_test': roc_auc_test,
                    }

                    results.append(metrics)


                except CustomException as e:
                    logger.error(f"Error training {model_name}: {e}")

        return pd.DataFrame(results)
    
class BestModelEvaluation:
    def __init__(self):
        pass
    
    @track_performance
    @staticmethod
    def evaluate(model_dict, param_dict, X_train, X_test, y_train, y_test, scoring):
        try:
            evaluator = ModelEvaluator(model_dict, param_dict, X_train, X_test, y_train, y_test, scoring)
            results_df = evaluator.evaluate_models()
    
            best_model_row = results_df.loc[results_df[scoring].idxmax()]
            best_model = best_model_row["Model"]
            best_score = best_model_row[scoring]
            logger.info(f"\nBest Model: {best_model} with Accuracy: {best_score}")
            return results_df

        except CustomException as ce:
            logger.error(f"Exception found: {ce}")

class SelectedModelsYaml:
    def __init__(self):
        pass
    
    @track_performance
    @staticmethod
    def tuned_models_yaml(models_dict:dict, params_dict:dict, metrics_df:pd.DataFrame, yaml_path:str, scoring:str):
        for short_name, model_instance in models_dict.items():
            model_name = type(model_instance).__name__

            if is_classifier(model_instance):

                try:
                    metrics_df_sorted = metrics_df.sort_values(by=scoring, ascending=False)
                    records = metrics_df_sorted.to_dict(orient='records')
                    yaml_models = []

                    for rec in records:
                        model_name = rec['Model']
                        model_entry = {
                                model_name: {
                                'model': models_dict[model_name].__class__.__name__,
                                'parameters': params_dict.get(model_name, {}),
                                'metrics': {
                                'accuracy_train': round(rec['accuracy_train'], 6),
                                'accuracy_test': round(rec['accuracy_test'], 6),
                                'precision_train': round(rec['precision_train'], 6),
                                'precision_test': round(rec['precision_test'], 6),
                                'recall_train': round(rec['recall_train'], 6),
                                'recall_test': round(rec['recall_test'], 6),
                                'f1_score_train': round(rec['f1_Score_train'], 6),
                                'f1_score_test': round(rec['f1_Score_test'], 6),
                                'roc_auc_train': round(rec['roc_auc_train'], 6),
                                'roc_auc_test': round(rec['roc_auc_test'], 6),
                            }
                        }
                    }
                        
                        yaml_models.append(model_entry)

                    merged_dict = {}
                    for entry in yaml_models:
                        merged_dict.update(entry)
                        
                    with open(yaml_path, "w") as file:
                        yaml.dump(merged_dict, file, sort_keys=False, default_flow_style=False, explicit_start=False)

                    logger.info(f"tuned_classifiers updated in yaml at: {yaml_path}")

                except CustomException as ce:
                    logger.error(f"Exception found: {ce}")

            
            elif is_regressor(model_instance):

                try:
                    metrics_df_sorted = metrics_df.sort_values(by='adj_r2_test', ascending=False)
                    records = metrics_df_sorted.to_dict(orient='records')
                    yaml_dict = {
                        rec['Model']: {
                            'model': models_dict[rec['Model']].__class__.__name__,
                            'parameters': params_dict.get(rec['Model'], {}),
                            'metrics': {
                                'r2_train': round(rec['r2_train'], 6),
                                'r2_test': round(rec['r2_test'], 6),
                                'adj_r2_train': round(rec['adj_r2_train'], 6),
                                'adj_r2_test': round(rec['adj_r2_test'], 6),
                                'mse_train': round(rec['mse_train'], 6),
                                'mse_test': round(rec['mse_test'], 6),
                                'rmse_train': round(rec['rmse_train'], 6),
                                'rmse_test': round(rec['rmse_test'], 6),
                                'mae_train': round(rec['mae_train'], 6),
                                'mae_test': round(rec['mae_test'], 6),
                            }
                        }
                        for rec in records
                    }

                    with open(yaml_path, "w") as file:
                        yaml.dump(yaml_dict, file, sort_keys=False)
                    logger.info(f"tuned_regressorss updated in yaml at: {yaml_path}")

                except CustomException as ce:
                    logger.error(f"Exception found: {ce}")
                  