# test_kafka.py

from kafka import KafkaProducer

try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: str(v).encode('utf-8')
    )
    future = producer.bootstrap_connected()
    print("Connected to Kafka:", future)
except Exception as e:
    print("Can't connect to Kafka:", e)