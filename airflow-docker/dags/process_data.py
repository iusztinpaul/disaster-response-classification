import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime

from data import process_data


def load(save_filepath: str = "./data/loaded.csv", **kwargs):
    ti = kwargs["ti"]

    df = process_data.load_data(
        messages_filepath="./data/disaster_messages.csv",
        categories_filepath="./data/disaster_categories.csv",
    )
    df.to_csv(save_filepath, index=False)

    ti.xcom_push(key="load_filepath", value=save_filepath)


def clean(save_filepath: str = "./data/cleaned.csv", **kwargs):
    ti = kwargs["ti"]

    df_file_path = ti.xcom_pull(key="load_filepath", task_ids="load_data")
    print("df_file_path: {}".format(df_file_path))
    df = pd.read_csv(df_file_path)
    df = process_data.clean_data(df)
    df.to_csv(save_filepath, index=False)

    ti.xcom_push(key="clean_filepath", value=save_filepath)


def save(save_filepath: str = "./data/data.db", **kwargs):
    ti = kwargs["ti"]

    df_file_path = ti.xcom_pull(key="clean_filepath", task_ids="clean_data")
    df = pd.read_csv(df_file_path)

    process_data.save_data(df, save_filepath)


with DAG(
        dag_id="ETL",
        start_date=datetime(2022, 7, 1, ),
        schedule_interval="@daily",
        catchup=False,
) as dag:
    load_data = PythonOperator(
        task_id="load_data",
        python_callable=load,
    )
    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=clean,
    )
    save_data = PythonOperator(
        task_id="save_data",
        python_callable=save,
    )

    load_data >> clean_data >> save_data
