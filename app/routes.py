# app/routes.py

import time
from datetime import datetime
from flask import Blueprint, render_template, request, send_file, jsonify, session
from werkzeug.utils import secure_filename
import os
import pydicom
import base64
import cv2
import numpy as np
from .dicom_validator import validate_dicom, clean_dicom
from .model_loader import load_model
from .inference import run_inference
from .config import MODEL_PATH, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from .kafka_producer import KafkaResultSender



main = Blueprint('main', __name__)
main.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'uploads'
RESULT_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, 'result_heatmap.jpg')
model = load_model(MODEL_PATH)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/upload', methods=['POST'])
def upload_dicom():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    base_name, ext = os.path.splitext(filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # --- Вызываем валидацию + анонимизацию ---
    is_valid, validation_result, extracted_tags = validate_dicom(filepath)

    # --- Если валидация вернула cleaned_file_path — используем его
    cleaned_filepath = os.path.join(UPLOAD_FOLDER, f"{base_name}_cleaned.dcm")

    if os.path.exists(cleaned_filepath):
        session['current_filepath'] = cleaned_filepath
    else:
        session['current_filepath'] = filepath  # fallback

    # --- Превью изображения ---
    try:
        ds = pydicom.dcmread(filepath)
        img_array = ds.pixel_array
        resized_img = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_LINEAR)
        display_img = ((resized_img - resized_img.min()) / (resized_img.max() - resized_img.min()) * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', display_img)
        preview_url = "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print("Ошибка при создании превью:", e)
        preview_url = None

    # --- Отправляем ответ ---
    warnings = validation_result if isinstance(validation_result, list) else []

    return jsonify({
        "preview_url": preview_url,
        "warnings": warnings,
        "extracted_tags": extracted_tags,
        "can_continue": True
    })


@main.route('/analyze', methods=['POST'])
def analyze_dicom():
    filepath = session.get('current_filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found or expired"}), 400

    try:
        start_time = time.time()
        result = run_inference(model, filepath)
        cv2.imwrite(RESULT_IMAGE_PATH, result["heatmap"])
        end_time = time.time()
        response_data = {
            "pathology_flag": result["pathology_flag"],
            "confidence_level": result["confidence_level"],
            "lesion_area_px": result["lesion_area_px"],
            "predicted_class": result["predicted_class"],
            "class_confidence": result["class_confidence"],
            "predicted_birads": result["predicted_birads"],
            "birads_confidence": result["birads_confidence"],
            "image_url": "/result_image?timestamp=" + str(int(time.time())),
            "processing_time_sec": f'{(end_time - start_time):.2f}'
        }

        # Сохраняем результат анализа в сессию
        session['analysis_result'] = response_data

        if result["lesion_area_mm2"] is not None:
            response_data["lesion_area_mm2"] = result["lesion_area_mm2"]

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@main.route('/result_image')
def get_result_image():
    return send_file(RESULT_IMAGE_PATH, mimetype='image/jpeg')

@main.route('/send_to_kafka', methods=['POST'])
def send_to_kafka():
    filepath = session.get('current_filepath')

    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "File not found or expired"}), 400

    try:
        ds = pydicom.dcmread(filepath)
        study_iuid = ds.StudyInstanceUID

    except Exception as e:
        return jsonify({"error": f"Can't read StudyInstanceUID from DICOM: {str(e)}"}), 500

    # Берём данные из сессии или предыдущего анализа
    analysis_result = session.get('analysis_result')
    if not analysis_result:
        return jsonify({"error": "No analysis result to send"}), 400

    # Инициализируем Kafka Producer
    sender = KafkaResultSender(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)

    # Формируем отчёт и заключение
    report = analysis_result['predicted_class']
    conclusion = 'Подозрение на патологию' if analysis_result['pathology_flag'] else 'Норма'

    # Отправляем в Kafka
    try:
        sent_message = sender.send_kafka_message(
            study_iuid,
            analysis_result['pathology_flag'],
            analysis_result['confidence_level'],
            report,
            conclusion
        )
        return jsonify({
            "status": "success",
            "message": "Sent to Kafka",
            "kafka_message": sent_message
        })
    except Exception as e:
        return jsonify({"error": f"Kafka send failed: {str(e)}"}), 500
    
@main.route('/generate_report', methods=['GET'])
def generate_report():
    analysis_result = session.get('analysis_result')
    if not analysis_result:
        return jsonify({"error": "No analysis result found"}), 400

    try:
        # Получаем StudyInstanceUID из DICOM
        filepath = session.get('current_filepath')
        ds = pydicom.dcmread(filepath)
        study_iuid = ds.StudyInstanceUID

        # Формируем отчёт
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        sr_report = {
            "Modality": "MG",
            "Область исследования": "Молочные железы",
            "Study Instance UID": study_iuid,
            "Дата и время заключения": now,
            "Наименование сервиса": "mammo-ai",
            "Версия сервиса": "1.0.3",
            "Назначение сервиса": "Автоматическая классификация патологий на маммограммах",
            "Предупреждение (1)": "Заключение подготовлено программным обеспечением с применением ИИ",
            "Предупреждение (2)": "Для исследовательских целей",
            "Технические данные": "Обработано: 1 изображение, размер: 512×512, глубина: 8 бит",
            "Описание": f"Выявлено образование типа {analysis_result['predicted_class']}, площадью ~{analysis_result['lesion_area_px']} px.",
            "Заключение": "Выявлены признаки возможного злокачественного новообразования" \
                if analysis_result['pathology_flag'] else "Патологий не выявлено",
            "Тип патологии": analysis_result['predicted_class'],
            "Вероятность патологии": f"{analysis_result['confidence_level']}%",
            "Рекомендации": "Требуется консультация маммолога и дообследование",
        }

        return jsonify({"report": sr_report})

    except Exception as e:
        return jsonify({"error": f"Can't generate report: {str(e)}"}), 500