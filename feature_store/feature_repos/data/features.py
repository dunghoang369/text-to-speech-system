from datetime import timedelta

from data_sources import speech_stats_batch_source, speech_stats_stream_source
from entities import speech
from feast import FeatureView, Field
from feast.stream_feature_view import stream_feature_view
from feast.types import String
from pyspark.sql import DataFrame

speech_stats_view = FeatureView(
    name="speech_stats",
    description="speech text",
    entities=[speech],
    ttl=timedelta(days=36500),
    schema=[
        Field(name="speech", dtype=String),
    ],
    online=True,
    source=speech_stats_batch_source,
)

@stream_feature_view(
    entities=[speech],
    ttl=timedelta(days=36500),
    mode="spark",
    schema=[
        Field(name="speech", dtype=String),
    ],
    timestamp_field="created",
    online=True,
    source=speech_stats_stream_source,
)
def speech_stats_stream(df: DataFrame):
    from pyspark.sql.functions import col

    return (
        df.withColumn("new_speech", col("speech") + "!")
        .drop(
            "speech",
        )
        .withColumnRenamed("new_speech", "speech")
    )

