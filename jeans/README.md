
## run redis server 
```docker run --name redis-server -p 6379:6379 -d redis```
To test ping do
```
sudo apt install redis-tools
redis-cli ping  # should return PONG
```