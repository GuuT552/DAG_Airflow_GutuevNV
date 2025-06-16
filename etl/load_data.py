import pandas as pd
import os
import logging
from ucimlrepo import fetch_ucirepo

def load_data():
    logging.basicConfig(level=logging.INFO)
    logging.info("Fetching dataset from UCI repository...")

    data = fetch_ucirepo(id=17)
    X = data.data.features
    y = data.data.targets

    os.makedirs('results', exist_ok=True)
    X.to_csv('results/X.csv', index=False)
    y.to_csv('results/y.csv', index=False)
    logging.info("Saved X and y to results/")