from kafka import KafkaProducer
import json
import time

# Сообщение для отправки
msg = {
    "studyIUID": "1.2.840.10008.5.1.4.1.1.7",
    "aiResult": {
        "seriesIUID": "1.2.840.10008.5.1.4.1.1.7.1",
        "pathologyFlag": True,
        "confidenceLevel": 92,
        "modelId": "mammo-unetpp",
        "modelVersion": "1.0",
        "norma": 0
    },
    "dateTimeParams": {
        "downloadStartDT": "2025-06-02T12:00:00Z",
        "processEndDT": "2025-06-02T12:01:00Z"
    }
}

# Создаём продюсер
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks=1,
    retries=5,
    linger_ms=10
)

# Отправляем сообщение
future = producer.send('mammo_results', value=msg)
try:
    result = future.get(timeout=10)
    print(f"✅ Сообщение отправлено: topic={result.topic}, partition={result.partition}, offset={result.offset}")
except Exception as e:
    print(f"❌ Ошибка при отправке: {e}")

# Завершаем работу продюсера
producer.close()
