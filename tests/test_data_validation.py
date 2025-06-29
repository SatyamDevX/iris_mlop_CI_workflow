import pandas as pd
from sklearn.datasets import load_iris

def test_data_shape():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    assert X.shape[1] == 4

def test_no_missing_values():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    assert X.isnull().sum().sum() == 0

