# test_consumer.py

from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'mammo_results',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    group_id='mammo-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print("Ожидаем сообщения...")

for message in consumer:
    print(f"\nПолучено сообщение из топика {message.topic}:\n")
    print(json.dumps(message.value, indent=2, ensure_ascii=False))