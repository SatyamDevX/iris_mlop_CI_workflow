import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier()
    clf.fit(X, y)
    joblib.dump(clf, 'model.pkl')
    print("Model trained and saved.")

if __name__ == "__main__":
    train_and_save()

