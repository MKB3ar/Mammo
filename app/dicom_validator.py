# app/dicom_validator.py

import pydicom
from datetime import datetime
import re
from app.config import ALLOWED_DICOM_TAGS


def validate_dicom(file_path):
    try:
        ds = pydicom.dcmread(file_path)
    except Exception as e:
        return False, f"Invalid DICOM file: {e}", {}

    errors = []
    extracted_tags = {}

    # --- Проверка Modality специфичная для маммографии ---
    modality = ds.data_element('Modality')
    if not modality:
        errors.append("Tag (0008, 0060) (Modality): required but missing")
    elif modality.value not in ('MG', 'OT'):
        errors.append("ВАЖНО!! Вы собираетесь проверять файл не маммографии")

    # --- Собираем теги для интерфейса ---
    tags_to_extract = {
        'SOP Class UID',
        'Content Date',
        'Content Time',
        "Secondary Capture Device Manufacturer's Model Name",
        'Secondary Capture Device Manufacturer',
        "Patient's Birth Date",
        "Patient's Sex",
        'Study Instance UID',
        'Modality',
        "Conversion Type"
    }

    for tag in ds:
        if tag.name in tags_to_extract:
            value = str(tag.value) if tag.value is not None else '(empty)'
            extracted_tags[tag.name] = value

        info = ALLOWED_DICOM_TAGS.get(tag.tag, {})
        value = tag.value if tag.value is not None else None

        # Проверка существования
        if info.get('required') and value is None:
            errors.append(f"Tag {tag.tag} ({tag.name}): required but missing")
            continue

        # Точное значение
        if 'required_value' in info:
            expected = info['required_value']
            if str(value) != expected:
                errors.append(f"Tag {tag.tag} ({tag.name}): expected '{expected}', got '{value}'")

        # Формат через regex
        if 'format' in info:
            pattern = info['format']
            if value is None or not re.match(pattern, str(value)):
                errors.append(f"Tag {tag.tag} ({tag.name}): invalid format, got '{value}'")

        # Числовые значения
        if 'min' in info or 'max' in info:
            if value is None:
                errors.append(f"Tag {tag.tag} ({tag.name}): required for min/max check, but missing")
                continue
            try:
                num_val = float(value)
                if 'min' in info and num_val <= info.get('min', float('-inf')):
                    errors.append(f"Tag {tag.tag} ({tag.name}): must be > {info['min']}, got {num_val}")
                if 'max' in info and num_val >= info.get('max', float('inf')):
                    errors.append(f"Tag {tag.tag} ({tag.name}): must be < {info['max']}, got {num_val}")
            except ValueError:
                errors.append(f"Tag {tag.tag} ({tag.name}): expected numeric value, got '{value}'")

        # Проверка списка чисел
        if info.get('type') == 'float_list':
            if value is None:
                errors.append(f"Tag {tag.tag} ({tag.name}): required for list check, but missing")
                continue
            parts = str(value).split('\\')
            if len(parts) != 2:
                errors.append(f"Tag {tag.tag} ({tag.name}): expected two values separated by \\")
            else:
                for part in parts:
                    try:
                        val = float(part)
                        if val <= info.get('min', 0):
                            errors.append(f"Tag {tag.tag}: value {part} is below minimum {info.get('min', 0)}")
                    except ValueError:
                        errors.append(f"Tag {tag.tag}: expected numeric value, got '{part}'")

        # Допустимые значения
        if 'allowed_values' in info and value is not None:
            if int(value) not in info['allowed_values']:
                errors.append(f"Tag {tag.tag} ({tag.name}): unexpected value {value}")

    # --- Анонимизация и сохранение очищенного файла ---
    from .dicom_validator import clean_dicom
    cleaned_ds = clean_dicom(ds)
    cleaned_file_path = file_path + ".cleaned.dcm"
    cleaned_ds.save_as(cleaned_file_path)

    if errors:
        return False, errors, extracted_tags

    return True, "DICOM metadata valid", extracted_tags

def clean_dicom(ds):
    """
    Очищает/удаляет чувствительные теги (например, имя пациента)
    """
    sensitive_tags = {
        "Patient's Name",  # "Patient's Name"
    }

    for tag_name in sensitive_tags:
        if ds.data_element(tag_name):
            print(f"[DICOM Clean] Удаляем тег {tag_name}")
            del ds.data_element(tag_name).value  # Можно использовать просто удаление элемента целиком
            # Или установить на всякий случай значение по умолчанию:
            # setattr(ds, tag_name, '***')
    
    return ds