import logging
import shutil
from datetime import datetime
import os

def save_results():
    logging.basicConfig(level=logging.INFO)
    logging.info("Archiving results...")

    date_str = datetime.today().strftime('%Y-%m-%d')
    target_dir = f"results/archive/{date_str}"
    os.makedirs(target_dir, exist_ok=True)

    shutil.copy("results/model.pkl", f"{target_dir}/model.pkl")
    shutil.copy("results/metrics.json", f"{target_dir}/metrics.json")

    logging.info(f"Results archived to {target_dir}")