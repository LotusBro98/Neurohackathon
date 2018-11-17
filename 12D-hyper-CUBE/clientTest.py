import requests as rq

res = rq.get("http://localhost:88")
print(res.text)