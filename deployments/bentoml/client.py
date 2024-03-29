import requests
import numpy as np
from scipy.io.wavfile import write

headers = {
    'accept': 'application/json',
    'content-type': 'application/x-www-form-urlencoded',
}

json_data = {
    'text': "love didn't meet her at her best, love meets het at her mess"}

response = requests.post('http://localhost:5000/detection', json=json_data, headers=headers)
audio = response.json()["result"]
audio = np.array(audio)
print(audio)
write(f"test_api.wav", 22050, audio)
