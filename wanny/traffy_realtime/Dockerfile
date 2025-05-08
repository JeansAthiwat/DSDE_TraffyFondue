FROM python:3.9-slim

# turn OFF implicit hash checking
ENV PIP_REQUIRE_HASHES=0

# OS build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev libsasl2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) click first
RUN pip install --no-cache-dir click==8.1.7

# 2) faust-streaming w/out deps so it keeps our click
RUN pip install --no-cache-dir faust-streaming==0.6.9 --no-deps

# 3) rest of Python deps (hashes now ignored)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4) copy source + models
COPY . .

CMD ["sleep", "infinity"]
