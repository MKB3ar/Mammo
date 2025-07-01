import pydicom
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch

from app.model_loader import load_model  # ваш файл с архитектурой и функцией загрузки весов

# --- Параметры ---
dicom_path = 'uploads\\20588562.dcm'      # путь к DICOM-изображению
mask_path = 'uploads\\20588562.png'        # путь к маске (DICOM или npy/png, если есть)
model_weights = 'models\\unetpp_model.pt' # путь к весам модели

# --- Загрузка DICOM-изображения ---
dicom = pydicom.dcmread(dicom_path)
image = dicom.pixel_array.astype(np.float32)
image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)  # нормализация [0,1]

# --- Загрузка маски из PNG ---
mask = np.array(Image.open(mask_path).convert('L'))  # grayscale
mask = (mask > 0).astype(np.float32)  # бинаризация

# --- Загрузка модели ---
model = load_model(model_weights)

# --- Подготовка изображения для модели ---
input_tensor = torch.from_numpy(image[None, None, ...]).float()  # [B, C, H, W]

# --- Получение предсказания ---
with torch.no_grad():
    pred_mask, _, _ = model(input_tensor)
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.float32)  # бинаризация

# --- Визуализация ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# Оригинал
axs[0].imshow(image, cmap='gray')
axs[0].set_title('DICOM изображение')
axs[0].axis('off')

# Маска, наложенная на изображение
axs[1].imshow(image, cmap='gray')
axs[1].imshow(mask, cmap='Reds', alpha=0.5)
axs[1].set_title('Маска (наложение)')
axs[1].axis('off')

# Результат модели
axs[2].imshow(image, cmap='gray')
axs[2].imshow(pred_mask, cmap='Blues', alpha=0.5)
axs[2].set_title('Модель (наложение)')
axs[2].axis('off')

# Оба наложения
axs[3].imshow(image, cmap='gray')
axs[3].imshow(mask, cmap='Reds', alpha=0.5)
axs[3].imshow(pred_mask, cmap='Blues', alpha=0.5)
axs[3].set_title('Маска и модель')
axs[3].axis('off')

# --- Легенда ---
legend_elements = [
    Patch(facecolor='red', edgecolor='r', label='Маска (оригинал)', alpha=0.5),
    Patch(facecolor='blue', edgecolor='b', label='Маска (модель)', alpha=0.5)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize='large', bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()