import faust, json, redis
import os
from fe.preprocessing import engineer_features
from config import settings

app = faust.App(
    "traffy-fe",
    broker=f"kafka://{settings.KAFKA_BROKER}",
    value_serializer="raw",
)

raw_topic = app.topic("raw-reports")          # JSON bytes
feat_topic = app.topic("features-reports", value_serializer="json")  # JSON dict

rds = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True)

@app.agent(raw_topic)
async def process(stream):
    async for msg in stream:
        raw = json.loads(msg)
        feats = engineer_features(raw)

        rid = feats["ticket_id"]
        rds.hset(f"feat:{rid}", mapping=feats)

        await feat_topic.send(value=feats)
