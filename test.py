import requests

headers = {
    "Content-Type": "application/json",
}

json_data = {"result": [1, 2, 3, 4]}

status = requests.post('https://f189-27-77-246-74.ngrok-free.app/process', json=json_data, headers=headers)

print(status)