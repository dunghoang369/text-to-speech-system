# This is an example feature definition file

from datetime import timedelta

from feast import KafkaSource
from feast.data_format import JsonFormat
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import \
    PostgreSQLSource

speech_stats_batch_source = PostgreSQLSource(
    name="data",
    query="SELECT * FROM data",
    timestamp_field="created",
)

speech_stats_stream_source = KafkaSource(
    name="speech_stats_stream_source",
    kafka_bootstrap_servers="localhost:9092",
    topic="speech_1",
    timestamp_field="created",
    batch_source=speech_stats_batch_source,
    message_format=JsonFormat(
        schema_json="created timestamp, speech_id integer, speech string"
    ),
    watermark_delay_threshold=timedelta(minutes=1),
)