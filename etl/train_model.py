import pandas as pd
import logging
import joblib
from sklearn.linear_model import LogisticRegression
import os

def train():
    logging.basicConfig(level=logging.INFO)
    logging.info("Training LogisticRegression model...")

    X_train = pd.read_csv('results/X_train.csv')
    y_train = pd.read_csv('results/y_train.csv').values.ravel()

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, 'results/model.pkl')
    logging.info("Model saved to results/model.pkl")