
import redis

rds = redis.Redis(host="localhost", port=6379, decode_responses=True)

keys = rds.keys("feat:*")
print(f"Found {len(keys)} keys to delete.")

if keys:
    rds.delete(*keys)
    print("Deleted all feat:* keys.")
else:
    print("No keys to delete.")
