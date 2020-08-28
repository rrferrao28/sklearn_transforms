from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')




class Balanceamento(BaseEstimator, TransformerMixin):
        def __init__(self, X, y):
            self.X = X
            self.y = y


        def fit(self, X, y):
            return self

        def transform(self, X, y):
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy='all')
            # Primeiro realizamos a cópia do dataframe 'X' de entrada
            self.X, self.y = smote.fit_resample(X,y)
            # Retornamos um novo dataframe sem as colunas indesejadas
            return self.X, self.y
