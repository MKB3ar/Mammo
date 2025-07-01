# app/kafka_producer.py

import json
from datetime import datetime
from kafka import KafkaProducer


class KafkaResultSender:
    def __init__(self, bootstrap_servers, topic):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic

    def send_kafka_message(self, study_iuid, pathology_flag, confidence_level, report, conclusion):
        message = {
            "studyIUID": study_iuid,
            "aiResult": {
                "seriesIUID": f"{study_iuid}.1",  # Примерный UID серии
                "pathologyFlag": pathology_flag,
                "confidenceLevel": confidence_level,
                "modelId": "mammo-unetpp",
                "modelVersion": "1.0",
                "report": report,
                "conclusion": conclusion,
                "norma": 0 if pathology_flag else 1
            },
            "dateTimeParams": {
                "downloadStartDT": datetime.utcnow().isoformat() + "Z",
                "downloadEndDT": datetime.utcnow().isoformat() + "Z",
                "processStartDT": datetime.utcnow().isoformat() + "Z",
                "processEndDT": datetime.utcnow().isoformat() + "Z"
            }
        }

        self.producer.send(self.topic, value=message)
        self.producer.flush()
        return message