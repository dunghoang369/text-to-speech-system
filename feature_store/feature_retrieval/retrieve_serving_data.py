import requests
import numpy as np
import pandas as pd
from datetime import datetime

from feast import FeatureStore
from scipy.io.wavfile import write

# Note: see https://docs.feast.dev/getting-started/concepts/feature-retrieval for
# more details on how to retrieve for all entities in the offline store instead
entity_df = pd.DataFrame.from_dict(
    {
        # entity's join key -> entity values
        "speech_id": [1, 2, 3, 4],
    }
)

entity_df["event_timestamp"] = datetime.now()
# print(entity_df["event_timestamp"])
print("-------------------->", entity_df["event_timestamp"])
# Initialize the feature store
store = FeatureStore(repo_path="../feature_repos/data")

# Get serving data from feature store
serving_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "speech_stats:speech",
    ],
).to_df()

print("----- Feature schema -----\n")
print(serving_df.info())

print()
print("----- Example features -----\n")
print(serving_df.head())

speech = serving_df["speech"].values
speech_id = serving_df["speech_id"].values

headers = {
    'accept': 'application/json',
    'content-type': 'application/x-www-form-urlencoded',
}

for i in range(len(speech_id)):
    json_data = {
        'text': speech[i]}

    response = requests.post('http://localhost:5000/detection', json=json_data, headers=headers)
    audio = response.json()["result"]
    audio = np.array(audio)
    write(f"output_wav/{speech_id[i]}.wav", 22050, audio)