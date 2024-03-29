import requests
import numpy as np
from scipy.io.wavfile import write

# We will send requests with content-type is json
headers = {
    "Content-Type": "application/json",
}

def docker_client():
    # Define our data for prediction
    json_data = {"Text": "Hello World"}

    response = requests.post('http://localhost:9000/predict', json=json_data, headers=headers)
    audio = response.json()["result"]
    audio = np.array(audio)
    write(f"test_api_docker.wav", 22050, audio)

def seldon_client():
    # Define our data for prediction
    json_data = {"data": {"Text": "Hello World"}}
    
    response = requests.post('http://localhost:8000/seldon/seldon/seldon-model-logging/api/v1.0/predictions', json=json_data, headers=headers)
    audio = response.json()["result"]
    audio = np.array(audio)
    write(f"test_api_seldon.wav", 22050, audio)

def kserve_client():
    # Define our data for prediction
    json_data = {"Text": "Hello World"}
    
    response = requests.post('http://tts.kserve-deployment.34.170.87.225.sslip.io/predict', json=json_data)
    audio = response.json()["result"]
    audio = np.array(audio)
    write(f"test_api_kserve.wav", 22050, audio)

if __name__ == "__main__":
    # docker_client()
    seldon_client()
    kserve_client()




