FROM python:3.9-slim

# Disable hash checking
ENV PIP_REQUIRE_HASHES=0
ENV PYTHONUNBUFFERED=1

# ---- OS build deps -------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libffi-dev libssl-dev libsasl2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 1) Pin click first -------------------------------------------
RUN pip install --no-cache-dir click==8.1.7

# ---- 2) Install faust-streaming without its deps -------------------
RUN pip install --no-cache-dir faust-streaming==0.6.8 --no-deps

# ---- 3) Rest of Python deps ---------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 4) Copy source & models --------------------------------------
COPY . .

# ---- 5) Default cmd (replace later with your real entrypoint) -----
CMD ["python", "fe/fe_app.py"]
