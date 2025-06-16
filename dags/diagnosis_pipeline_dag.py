from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import sys
sys.path.append('/opt/airflow/etl')

from load_data import load_data
from preprocess import preprocess
from train_model import train
from evaluate import evaluate
from save_results import save_results

default_args = {
    'start_date': datetime(2024, 1, 1),
}

with DAG('diagnosis_pipeline',
         schedule_interval=None,
         catchup=False,
         default_args=default_args,
         tags=['medical']) as dag:

    t1 = PythonOperator(task_id='load_data', python_callable=load_data)
    t2 = PythonOperator(task_id='preprocess', python_callable=preprocess)
    t3 = PythonOperator(task_id='train_model', python_callable=train)
    t4 = PythonOperator(task_id='evaluate_model', python_callable=evaluate)
    t5 = PythonOperator(task_id='save_results', python_callable=save_results)

    t1 >> t2 >> t3 >> t4 >> t5
