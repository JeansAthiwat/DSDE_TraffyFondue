# DSDE_TraffyFondue
Locked In

# HOW TO USE
1. install required dependencies.

2. start the services:
    at "~/wanny/traffy_realtime"
    run ```docker compose up --build fe_app``` 
    to start the data pipeline services (kafka, redis, zookeeper, fe_app)

    at "~/wanny/traffy_realtime"
    run ```docker compose exec fe_app python ingest/producer.py```  
    to simulate new task ingestions.

    at "~/wanny/traffy_realtime"
    run ```uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000``` 
    to start the Lgb model services.

3. Start Streamlit

    at "~/wanny/traffy_realtime/streamlit"
    run ```streamlit run app.py```
    to run the web app.
    