from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis, joblib
import pandas as pd

# load both pipelines once
lgb_inprog  = joblib.load("models/lgb_pipeline_inprog.joblib")
lgb_finish  = joblib.load("models/lgb_pipeline_finish.joblib")

rds = redis.Redis(host="redis", port=6379, decode_responses=True)
app = FastAPI(title="Traffy Predictor")

class TicketRequest(BaseModel):
    ticket_id: str

@app.post("/predict")
def predict(req: TicketRequest):
    hkey = f"feat:{req.ticket_id}"
    feats = rds.hgetall(hkey)
    if not feats:
        raise HTTPException(404, "features not found")

    # cast numeric fields back to correct dtypes
    df = pd.DataFrame([{
        "text":     feats["text"],
        "hour":     int(feats["hour"]),
        "weekday":  int(feats["weekday"]),
        "grid_x":   int(feats["grid_x"]),
        "grid_y":   int(feats["grid_y"]),
        "zone_id":  feats["zone_id"],
    }])

    pred_in  = float(lgb_inprog.predict(df)[0])
    pred_fin = float(lgb_finish.predict(df)[0])

    return {
        "ticket_id": req.ticket_id,
        "start_in_minutes": round(pred_in, 1),
        "resolve_minutes":  round(pred_fin, 1)
    }
