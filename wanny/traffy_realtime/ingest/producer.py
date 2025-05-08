import os, json, uuid, datetime, random
from kafka import KafkaProducer

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")     # default = in-container
if "." in os.getenv("COMPUTERNAME", ""):             # crude “am I on Windows?”
    BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")

TOPIC  = os.getenv("KAFKA_TOPIC", "raw-reports")

producer = KafkaProducer(
    bootstrap_servers=BROKER,
    value_serializer=lambda d: json.dumps(d, ensure_ascii=False).encode()
)

msg = {
    "ticket_id": str(uuid.uuid4()),
    "comment":   random.choice(["ทางเท้าชำรุด", "ไฟถนนดับ", "ถังขยะล้น"]),
    "type":      random.choice(["ถนน/ทางเท้า", "ไฟฟ้าและแสงสว่าง"]),
    "coords":    "100.60802,13.80367",
    "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
}

producer.send(TOPIC, msg).get(timeout=10)
producer.flush()
print("pushed", msg["ticket_id"])
