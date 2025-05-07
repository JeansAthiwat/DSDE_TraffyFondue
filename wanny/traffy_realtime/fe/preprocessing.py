import re, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from joblib import load

CELL_SIZE = 0.01  # same as training
# load your pre-fitted kmeans centroids (from the notebook)
KMEANS = load("models/kmeans_zone.joblib")  # <-- save this from training time

def engineer_features(raw: dict) -> dict:
    """
    Turn a raw Traffy event into the six-column feature row
    needed by the LightGBM pipelines.
    """
    typ   = raw.get("type", "")
    comm  = raw.get("comment", "")
    text  = f"{typ} {comm}"
    ts    = pd.to_datetime(raw["timestamp"])
    hour, weekday = ts.hour, ts.dayofweek

    lon, lat = map(float, raw["coords"].split(","))
    grid_x, grid_y = int(lon // CELL_SIZE), int(lat // CELL_SIZE)
    zone_id = str(int(KMEANS.predict([[lon, lat]])[0]))

    features = {
        "ticket_id": raw["ticket_id"],
        "text": text,
        "hour": hour,
        "weekday": weekday,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "zone_id": zone_id,
    }
    return features
