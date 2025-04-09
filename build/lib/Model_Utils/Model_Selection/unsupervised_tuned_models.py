import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.base import ClusterMixin
from scipy.spatial.distance import cdist
from Common_Utils import logger, CustomException, track_performance

class SearchStrategy:
    def search(self, model_name, model, X_train, param_grid):
        pass

class GridSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, param_grid):
        try:
            logger.info(f"Starting GridSearchCV for hyper_param tuning..")
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train)
            return grid_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in GridSearchCV: {e}")
        


class RandomSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, param_grid):
        try:
            logger.info(f"Starting RandomizedSearchCV for hyper_param tuning..")
            random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42)
            random_search.fit(X_train)
            return random_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in RandomizedSearchCV: {e}")

class BayesianSearchStrategy(SearchStrategy):
    def search(self, model_name, model, X_train, param_grid):
        try:
            logger.info(f"Starting BayesSearchCV for hyper_param tuning..")
            bayes_search = BayesSearchCV(model, param_grid, n_iter=10, cv=5, random_state=42)
            bayes_search.fit(X_train)
            return bayes_search.best_params_
        
        except CustomException as e:
            logger.error(f"Error in BayesSearchCV: {e}")

class UnsupervisedSearchStrategy(SearchStrategy):
    """Custom hyperparameter tuning for unsupervised models."""
    
    def search(self, model_name, model, X_train, param_grid):
        best_params = {}
        
        if isinstance(model, KMeans):
            best_score = float('inf')
            for n_clusters in param_grid.get('n_clusters', [3]):
                for init in param_grid.get('init', ['k-means++']):
                    for max_iter in param_grid.get('max_iter', [300]):
                        km = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
                        km.fit(X_train)
                        inertia = km.inertia_  # Sum of squared distances to closest cluster center
                        if inertia < best_score:
                            best_score = inertia
                            best_params = {'n_clusters': n_clusters, 'init': init, 'max_iter': max_iter}
        
        elif isinstance(model, DBSCAN):
            best_score = -1
            for eps in param_grid.get('eps', [0.5]):
                for min_samples in param_grid.get('min_samples', [5]):
                    db = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = db.fit_predict(X_train)
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Ignore noise points
                    if num_clusters > best_score:
                        best_score = num_clusters
                        best_params = {'eps': eps, 'min_samples': min_samples}
        
        elif isinstance(model, AgglomerativeClustering):
            best_score = -1
            for n_clusters in param_grid.get('n_clusters', [3]):
                for linkage in param_grid.get('linkage', ['ward']):
                    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                    labels = ac.fit_predict(X_train)
                    num_clusters = len(set(labels))
                    if num_clusters > best_score:
                        best_score = num_clusters
                        best_params = {'n_clusters': n_clusters, 'linkage': linkage}
        
        elif isinstance(model, PCA) or isinstance(model, TruncatedSVD):
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train)
            best_params = grid_search.best_params_
        
        elif isinstance(model, TSNE):
            best_score = float('inf')
            for n_components in param_grid.get('n_components', [2]):
                for perplexity in param_grid.get('perplexity', [30]):
                    for learning_rate in param_grid.get('learning_rate', [200]):
                        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)
                        transformed = tsne.fit_transform(X_train)
                        dispersion = np.mean(cdist(transformed, transformed))  # Average distance between points
                        if dispersion < best_score:
                            best_score = dispersion
                            best_params = {'n_components': n_components, 'perplexity': perplexity, 'learning_rate': learning_rate}

        return best_params


class HyperparameterTuner:
    def __init__(self, models, X_train, yaml_file, search_strategy):
        self.models = models
        self.X_train = X_train
        self.hyperparams = self.load_hyperparameters(yaml_file)
        self.search_strategy = search_strategy
        self.best_params = {}

    @staticmethod
    def load_hyperparameters(yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)

    def tune_model(self, model_name, model):
        param_grid = self.hyperparams.get('unsupervised_models', {}).get(model_name, {})
        return self.search_strategy.search(model_name, model, self.X_train, param_grid)

    def tune_models(self):
        for model_name, model in self.models.items():
            logger.info(f"Tuning model: {model_name}")
            try:
                self.best_params[model_name] = self.tune_model(model_name, model)
            except CustomException as e:
                logger.error(f"Error tuning {model_name}: {e}")
        return self.best_params
    
class UnsupervisedHyperparameterSearchMethods:
    def __init__(self):
        pass
        

    methods = { "grid_search_cv":GridSearchStrategy(),
                "random_search_cv": RandomSearchStrategy(),
                "bayesian_search_cv": BayesianSearchStrategy(),
                
                }
    
    @track_performance
    @staticmethod
    def tuned_model_parameters(models= None , X = None, yaml_file:str = None , chosen_strategy:str = None ):
        try:
            if chosen_strategy in UnsupervisedHyperparameterSearchMethods.methods:
                tuner = HyperparameterTuner(models,X, yaml_file, UnsupervisedHyperparameterSearchMethods.methods[chosen_strategy] )
                best_hyperparameters = tuner.tune_models()
                logger.info(f'"the best hyperparameters for unsupervised models provided by {chosen_strategy} is: {best_hyperparameters}" ')
                return best_hyperparameters
            
        except CustomException as ce:
            logger.error(f"Exception found: {ce}")

        