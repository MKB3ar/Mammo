# app/config.py
import os

KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'mammo_results'


# Получаем путь к текущей директории
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'unetpp_model.pt')

ALLOWED_DICOM_TAGS = {
    (0x0008, 0x0060): {'name': 'Modality', 'required_value': 'MG'},
    (0x0010, 0x0010): {'name': 'PatientName', 'required': True},
    (0x0010, 0x0020): {'name': 'PatientID', 'required': True},
    (0x0010, 0x0030): {'name': 'PatientBirthDate', 'format': r'^\d{8}$'},
    (0x0018, 0x1151): {'name': 'XRayTubeCurrent', 'min': 0.1},
    (0x0018, 0x0060): {'name': 'KVP', 'min': 20, 'max': 40},
    (0x0018, 0x1164): {'name': 'ImagerPixelSpacing', 'type': 'float_list', 'min': 0.01},
    (0x0028, 0x0010): {'name': 'Rows', 'min': 1},
    (0x0028, 0x0011): {'name': 'Columns', 'min': 1},
    (0x0028, 0x0030): {'name': 'PixelSpacing', 'type': 'float_list', 'min': 0.01},
    (0x0028, 0x0100): {'name': 'BitsAllocated', 'allowed_values': [8, 12, 16]},
    (0x0028, 0x0101): {'name': 'BitsStored', 'max_tag': (0x0028, 0x0100)},
    (0x0028, 0x0103): {'name': 'PixelRepresentation', 'allowed_values': [0, 1]},
    (0x0020, 0x000D): {'name': 'StudyInstanceUID', 'required': True},
    (0x0020, 0x000E): {'name': 'SeriesInstanceUID', 'required': True},
    (0x0008, 0x103E): {'name': 'SeriesDescription', 'required': True},
    (0x0008, 0x0020): {'name': 'StudyDate', 'format': r'^\d{8}$'},
}