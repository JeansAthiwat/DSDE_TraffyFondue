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

for _ in range(3):
    msg = {
        "ticket_id": str(uuid.uuid4()),
        "comment":   random.choice(["ถังขยะไม่พอใช้", "น้ำขังบนถนน", "ไฟถนนกะพริบ", "ไฟถนนดับ", "ขยะเต็มถัง"]),
        "type":      random.choice(["ขยะ", "น้ำท่วม", "ไฟฟ้าและแสงสว่าง", "ถนน/ทางเท้า"]),
        "coords":    random.choice(["100.500,13.750", "100.589,13.811", "100.634,13.727", "100.611,13.879"]),
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
    }
    producer.send(TOPIC, msg).get(timeout=10)
    print("pushed", msg["ticket_id"])

producer.flush()