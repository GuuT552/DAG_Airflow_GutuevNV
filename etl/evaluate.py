import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

def evaluate():
    logging.basicConfig(level=logging.INFO)
    logging.info("Evaluating model...")

    X_test = pd.read_csv('results/X_test.csv')
    y_test = pd.read_csv('results/y_test.csv').values.ravel()
    model = joblib.load('results/model.pkl')

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='M'),
        "recall": recall_score(y_test, y_pred, pos_label='M'),
        "f1_score": f1_score(y_test, y_pred, pos_label='M')
    }

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info("Evaluation complete. Metrics saved.")