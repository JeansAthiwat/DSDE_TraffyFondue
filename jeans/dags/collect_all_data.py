from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# Import your scraping functions from pipeline
from pipeline.fetch_holiday import fetch_holiday
from pipeline.fetch_weather import fetch_weather
from pipeline.fetch_air_quality import fetch_air_quality

with DAG(
    dag_id="dsde_scraping_concurrent",
    start_date=days_ago(1),
    schedule_interval="@once",  # Run manually for testing
    catchup=False,
    tags=["dsde", "scraping", "redis"],
) as dag:

    start = EmptyOperator(task_id="start_task")

    # Your 3 scraping tasks
    holiday_task = PythonOperator(
        task_id="fetch_holiday", python_callable=fetch_holiday
    )

    weather_task = PythonOperator(
        task_id="fetch_weather", python_callable=fetch_weather
    )

    air_task = PythonOperator(
        task_id="fetch_air_quality", python_callable=fetch_air_quality
    )

    end = EmptyOperator(task_id="end_task")

    # Define parallel branches: start --> [holiday, weather, air] --> end
    start >> [holiday_task, weather_task, air_task] >> end
