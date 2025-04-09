import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from Common_Utils import logger, CustomException, track_performance
from sklearn.base import is_classifier, is_regressor
from abc import ABC, abstractmethod

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, model_name, model, X_train, y_train, param_grid, scoring):
        pass

class GridSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, y_train, param_grid, scoring):
        try:
            logger.info(f"Starting GridSearchCV for hyper_param tuning..")
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring)
            grid_search.fit(X_train, y_train)
            return grid_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in GridSearchCV: {e}")
    

class RandomSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, y_train, param_grid, scoring):
        try:
            logger.info(f"Starting RandomizedSearchCV for hyper_param tuning..")
            random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring=scoring, random_state=42)
            random_search.fit(X_train, y_train)
            return random_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in RandomizedSearchCV: {e}")

class BayesianSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, y_train, param_grid, scoring):
        try:
            logger.info(f"Starting BayesSearchCV for hyper_param tuning..")
            bayes_search = BayesSearchCV(model, param_grid, n_iter=10, cv=5, scoring=scoring, random_state=42)
            bayes_search.fit(X_train, y_train)
            return bayes_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in BayesSearchCV: {e}")



class HyperparameterTuner:
    def __init__(self, models, X_train, y_train, scoring, yaml_file, search_strategy):
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.hyperparams = self.load_hyperparameters(yaml_file)
        self.search_strategy = search_strategy
        self.best_params = {}

    @staticmethod
    def load_hyperparameters(yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def tune_model(self, model_name, model_or_fn):
        param_grid = self.hyperparams.get(model_name, {})
        return self.search_strategy.search(model_name, model_or_fn, self.X_train, self.y_train, param_grid, self.scoring)

    def tune_models(self,scoring):
        for model_name, model_or_fn in self.models.items():
            logger.info(f"Tuning model: {model_name}")
            try:
                if is_classifier(model_or_fn):
                    logger.info(f"{model_name} is a classification model.")
                    self.scoring = scoring
                elif is_regressor(model_or_fn):
                    logger.info(f"{model_name} is a regression model.")
                    self.scoring = scoring
                self.best_params[model_name] = self.tune_model(model_name, model_or_fn)
            except CustomException as e:
                logger.error(f"Error tuning {model_name}: {e}")
        return self.best_params
    
class SupervisedHyperparameterSearchMethods:
    def __init__(self):
        pass
        

    methods = { "grid_search_cv":GridSearchStrategy(),
                "random_search_cv": RandomSearchStrategy(),
                "bayesian_search_cv": BayesianSearchStrategy(),
                
                }
    
    @track_performance
    @staticmethod
    def tuned_model_parameters(models = None , X_train = None, y_train = None , scoring:str = None, yaml_file:str = None, chosen_strategy:str = None ):
        try:
            if chosen_strategy in SupervisedHyperparameterSearchMethods.methods:
                tuner = HyperparameterTuner(models, X_train, y_train, scoring, yaml_file ,SupervisedHyperparameterSearchMethods.methods[chosen_strategy] )
                best_hyperparameters = tuner.tune_models(scoring)
                logger.info(f"the best hyperparameters for supervised models provided by {chosen_strategy} is: {best_hyperparameters}")
                return best_hyperparameters
        
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")

