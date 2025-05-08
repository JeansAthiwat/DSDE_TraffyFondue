import faust, json, redis
import os
from fe.preprocessing import engineer_features

app = faust.App(
    "traffy-fe",
    broker=f"kafka://{os.getenv('KAFKA_BROKER','localhost:9092')}",
    value_serializer="raw",
)

raw_topic   = app.topic("raw-reports")          # JSON bytes
feat_topic  = app.topic("features-reports")     # JSON dict
rds         = redis.Redis(host="redis", port=6379, decode_responses=True)

@app.agent(raw_topic)
async def process(stream):
    async for msg in stream:
        raw = json.loads(msg)
        feats = engineer_features(raw)

        # persist to Redis hash for low-latency lookup by model API
        rid = feats["ticket_id"]
        rds.hset(f"feat:{rid}", mapping=feats)

        # publish to downstream topic for bulk consumers
        await feat_topic.send(value=feats)
