from feast import Entity

speech = Entity(
    name="speech",
    join_keys=["speech_id"],
    description="speech id",
)
