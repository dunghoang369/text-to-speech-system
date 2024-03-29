from pprint import pprint

from feast import FeatureStore

# Initialize the feature store
store = FeatureStore(repo_path="../feature_repos/data")
# Get serving data from feature store, we retrieve
# all features
feature_vector = store.get_online_features(
    features=[
        "speech_stats_stream:speech",
    ],
    entity_rows=[
        {"speech_id": 1},
    ],
).to_dict()

pprint(feature_vector)
