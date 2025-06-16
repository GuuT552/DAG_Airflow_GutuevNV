import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def preprocess():
    logging.basicConfig(level=logging.INFO)
    logging.info("Preprocessing data...")

    X = pd.read_csv('results/X.csv')
    y = pd.read_csv('results/y.csv')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train).to_csv('results/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('results/X_test.csv', index=False)
    y_train.to_csv('results/y_train.csv', index=False)
    y_test.to_csv('results/y_test.csv', index=False)
    logging.info("Saved train/test splits to results/")