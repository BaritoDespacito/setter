import requests

resp = requests.post(
    'http://127.0.0.1:5000/generate',
    json={"grade": 5, "angle": 40}
)

print(resp.blob())