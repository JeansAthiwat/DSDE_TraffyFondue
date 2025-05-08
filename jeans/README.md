TODO:

## run redis server 

```docker run --name redis-server -p 6379:6379 -d redis```

--name redis-server: Gives your container a friendly name.

-p 6379:6379: Exposes Redis on your local port.

-d: Runs in background (detached).

To test ping do
```
sudo apt install redis-tools
redis-cli ping  # should return PONG
```

## Pip i did (cant remember shit)
```
pip install apache-airflow redis beautifulsoup4 requests
sudo pip install apache-airflow
```