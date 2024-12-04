import cv2
import pytesseract
import torch
from pathlib import Path
import numpy as np

# Clase para detección con YOLOv5
class YOLOv5:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.5):
        """
        Inicializa el modelo YOLOv5 para detección.
        :param model_path: Ruta al modelo YOLOv5 personalizado o preentrenado.
        :param conf_threshold: Umbral de confianza para detecciones.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = conf_threshold  # Ajuste del umbral de confianza

    def detect(self, image_path):
        """
        Realiza detección en una imagen.
        :param image_path: Ruta de la imagen.
        :return: Resultados de detección como DataFrame.
        """
        results = self.model(image_path)
        return results.pandas().xyxy[0]  # DataFrame con las detecciones

# Función para realizar OCR con Tesseract
def apply_ocr(image):
    """
    Realiza OCR en una región de imagen.
    :param image: Región de la imagen en formato numpy array.
    :return: Texto extraído de la región.
    """
    # Preprocesamiento de imagen para mejorar OCR
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR con Tesseract
    config = "--oem 3 --psm 6"  # OEM 3: Default, PSM 6: Assumes a single uniform block of text
    text = pytesseract.image_to_string(binary_image, lang='eng', config=config)
    return text.strip()

# Pipeline principal para detección y OCR
def process_images(image_folder, yolov5_model_path):
    """
    Procesa imágenes para detección y OCR.
    :param image_folder: Carpeta con imágenes a procesar.
    :param yolov5_model_path: Ruta al modelo YOLOv5 personalizado.
    """
    # Inicializar el detector YOLOv5
    yolo = YOLOv5(model_path=yolov5_model_path)

    # Crear carpeta de salida
    output_folder = Path("./output")
    output_folder.mkdir(exist_ok=True)

    # Procesar cada imagen
    for image_path in Path(image_folder).glob("*.jpg"):
        print(f"Procesando: {image_path}")

        # Detección con YOLOv5
        detections = yolo.detect(str(image_path))
        image = cv2.imread(str(image_path))

        for i, detection in detections.iterrows():
            # Coordenadas de la región detectada
            x_min, y_min, x_max, y_max = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            label = detection['name']
            confidence = detection['confidence']

            # Extraer y procesar la región detectada
            cropped_image = image[y_min:y_max, x_min:x_max]
            text_detected = apply_ocr(cropped_image)

            print(f"[{label}] Confianza: {confidence:.2f}, Texto detectado: {text_detected}")

            # Dibujar resultados en la imagen original
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, text_detected, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Guardar la imagen con resultados
        output_path = output_folder / f"{image_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), image)
        print(f"Resultados guardados en: {output_path}")

if __name__ == "__main__":
    # Configuración de rutas
    image_folder = "./imagenes_etiquetas"  # Carpeta con imágenes de etiquetas
    yolov5_model_path = "./modelos/yolov5s.pt"  # Modelo YOLOv5 personalizado

    # Ejecutar el pipeline
    process_images(image_folder, yolov5_model_path)
