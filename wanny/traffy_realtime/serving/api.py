from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis, joblib
import pandas as pd
from config import settings

lgb_inprog  = joblib.load("./models/full_lgb_pipeline_inprog.joblib")
lgb_finish  = joblib.load("./models/full_lgb_pipeline_finished.joblib")

rds = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

app = FastAPI(title="Traffy Predictor")

class TicketRequest(BaseModel):
    ticket_id: str

@app.post("/predict")
def predict(req: TicketRequest):
    hkey = f"feat:{req.ticket_id}"
    feats = rds.hgetall(hkey)
    print("HKEY:", hkey)
    print("FEATS:", feats)
    if not feats:
        raise HTTPException(404, "features not found")
    print("FEATS:", feats)


    df = pd.DataFrame([{
    "text":     feats.get("text", ""),
    "hour":     int(feats.get("hour", 0)),
    "weekday":  int(feats.get("weekday", 0)),
    "grid_x":   int(feats.get("grid_x", 0)),
    "grid_y":   int(feats.get("grid_y", 0)),
    "zone_id":  feats.get("zone_id", "unknown"),
}])


    pred_in = float(lgb_inprog.predict(df)[0])
    pred_fin = float(lgb_finish.predict(df)[0])

    return {
        "ticket_id": req.ticket_id,
        "start_in_minutes": round(pred_in, 1),
        "resolve_minutes": round(pred_fin, 1)
    }
