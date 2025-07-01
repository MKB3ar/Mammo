# app/inference.py

import torch
import numpy as np
import cv2
from pydicom import dcmread
from PIL import Image
import os


def apply_colormap(mask, image, alpha=0.5):
    """
    Накладывает цветовую карту на изображение
    """
    # Нормализуем маску к диапазону [0, 255]
    heatmap = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Применяем цветовую карту
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Преобразуем исходное изображение в RGB, если нужно
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        image = (image * 255).astype(np.uint8)

    # Смешиваем оригинальное изображение и тепловую карту
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed


def preprocess_dicom(dicom_path, target_size=(512, 512)):
    ds = dcmread(dicom_path)
    img = ds.pixel_array

    # Если signed, то преобразуем
    if ds.PixelRepresentation == 1:
        img = img.astype(np.int16)

    # Нормализуем к 0..1
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Рескейлим под размер модели
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    return img_tensor, img_resized


# app/inference.py
CLASS_NAMES = ['CALC', 'CIRC', 'MISC', 'NORM']
BIRADS_LABELS = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5', 'BI-RADS 6']


def run_inference(model, dicom_path):
    input_tensor, raw_image = preprocess_dicom(dicom_path)

    with torch.no_grad():
        seg_mask, logits_cls, logits_br = model(input_tensor)

    seg_mask = seg_mask.squeeze().cpu().numpy()
    heatmap = apply_colormap(seg_mask, raw_image)

    # Предсказание класса (CALC, CIRC и т.д.)
    probs_cls = torch.softmax(logits_cls, dim=1).squeeze().cpu().numpy()
    pred_cls_idx = int(probs_cls.argmax())
    pred_class = CLASS_NAMES[pred_cls_idx]
    confidence_cls = int(probs_cls[pred_cls_idx] * 100)

    # Предсказание BI-RADS
    probs_br = torch.sigmoid(logits_br).squeeze().cpu().numpy()
    pred_br_idx = int((probs_br > 0.5).sum() + 1)  # [0,1,2,3,4,5] → BI-RADS 1–6
    pred_birads = CLASS_NAMES[pred_br_idx] if pred_br_idx < len(CLASS_NAMES) else "BI-RADS 6"
    confidence_br = int(100 - probs_br.mean() * 100)

    # Площадь поражения
    threshold = 0.3
    lesion_pixels = seg_mask > threshold
    lesion_area_px = np.sum(lesion_pixels)
    try:
        ds = dcmread(dicom_path)
        pixel_spacing_row = ds.get((0x0028, 0x0030), None)
        pixel_spacing = float(pixel_spacing_row.value[0]) if pixel_spacing_row else 1.0
        lesion_area_mm2 = lesion_area_px * (pixel_spacing ** 2)
    except Exception as e:
        print("Can't read PixelSpacing from DICOM:", e)
        lesion_area_mm2 = None

    pathology_flag = lesion_area_px > 100
    confidence_level = int(np.mean(seg_mask[lesion_pixels] * 100)) if lesion_pixels.any() else 0
    print(confidence_br)
    print(pred_birads)    

    return {
        "raw_image": raw_image,
        "heatmap": heatmap,
        "pathology_flag": bool(pathology_flag),
        "confidence_level": int(confidence_level),
        "lesion_area_px": int(lesion_area_px),
        "lesion_area_mm2": round(float(lesion_area_mm2), 2) if lesion_area_mm2 is not None else None,
        "predicted_class": pred_class,
        "class_confidence": confidence_cls,
        "predicted_birads": pred_birads,
        "birads_confidence": confidence_br
    }