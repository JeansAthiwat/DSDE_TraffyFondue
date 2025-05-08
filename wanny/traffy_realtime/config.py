import os

class Settings:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")

settings = Settings()
