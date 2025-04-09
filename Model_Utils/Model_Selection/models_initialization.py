from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from xgboost import XGBRegressor, XGBClassifier



class ModelInitializer:
    """Class to initialize Regression, Classification, and Unsupervised Learning models as static variables."""


    # Static variables for Regression Models
    regressor = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "XGBRegressor": XGBRegressor(),
        "SVR": SVR(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "MLPRegressor": MLPRegressor(),
         }

    # Static variables for Classification Models
    classifier = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
        "XGBClassifier": XGBClassifier(),
        "SVC": SVC(probability=True),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "MLPClassifier": MLPClassifier(),
        }

    # Static variables for Unsupervised Learning Models
    unsupervised_model = {
        "KMeans": KMeans(n_clusters=3),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
        "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
        "PCA": PCA(n_components=2),
        "TruncatedSVD": TruncatedSVD(n_components=2),
        "TSNE": TSNE(n_components=2)
    }

class SelectedModelInitializer:

    regressor = {
        "LinearRegression": ModelInitializer.regressor["LinearRegression"],
        "Ridge": ModelInitializer.regressor["Ridge"],
        "Lasso": ModelInitializer.regressor["Lasso"],
        }
    
    classifier = {
        "LogisticRegression": ModelInitializer.classifier["LogisticRegression"],
        "DecisionTreeClassifier": ModelInitializer.classifier["DecisionTreeClassifier"],
       
        }
    
    unsupervised_model = {
        "KMeans": ModelInitializer.unsupervised_model["KMeans"],
        "DBSCAN": ModelInitializer.unsupervised_model["DBSCAN"],
        "AgglomerativeClustering": ModelInitializer.unsupervised_model["AgglomerativeClustering"],
        }
