import hydra
import torch
import cv2
from random import randint
import numpy as np
import pandas as pd
from sort import *
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import os
from datetime import datetime, timedelta

# Lista de clases de vehículos a detectar (basada en COCO dataset)
# Índices: 2-car, 3-motorcycle, 5-bus, 7-truck
VEHICLE_CLASSES = [2, 3, 5, 7]  

# Variables globales
tracker = None
vehicle_data = []  # Almacenará datos para exportar a Excel
base_datetime = None  # Fecha base extraída del nombre del archivo
fps = 30.0  # Frames por segundo (valor predeterminado)
frame_count = 0  # Contador global de frames

def init_tracker():
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

rand_color_list = []
    
def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        # Obtener ID y categoría del vehículo
        id = int(identities[i]) if identities is not None else 0
        category_id = int(categories[i])
        category_name = names[category_id] if names is not None else str(category_id)
        
        # Centro del objeto
        box_center = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
        
        # Etiqueta con ID y tipo de vehículo
        label = f"{id} {category_name}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Dibujar rectángulo y texto
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
        
    return img

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

def extract_base_datetime(filename):
    """Extrae la fecha y hora base del nombre del archivo de video"""
    global base_datetime
    
    try:
        parts = filename.split('_')
        
        # Formato: 20210121_20210121081120_20210121081207_081119
        # Usar el tercer segmento (índice 2) que contiene el tiempo final del video
        if len(parts) >= 3 and len(parts[2]) >= 14:
            date_str = parts[2]
            # Formato: YYYYMMDDHHMMSS (20210121081207)
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(date_str[8:10])
            minute = int(date_str[10:12])
            second = int(date_str[12:14])
            
            base_datetime = datetime(year, month, day, hour, minute, second)
            print(f"Fecha base extraída: {base_datetime}")
            return True
    except Exception as e:
        print(f"Error al extraer fecha base: {e}")
    
    return False

def calculate_timestamp(frame_number):
    """Calcula el timestamp basado en el número de frame y el FPS"""
    global base_datetime, fps
    
    if base_datetime is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Como el timestamp base es el tiempo final del video,
    # calculamos el tiempo según la posición relativa del frame
    # asumiendo que frame_count es el total de frames procesados hasta ahora
    
    # Si solo procesamos frames impares, ajustamos el cálculo
    actual_frames_total = frame_count * 2  # estimación del total incluyendo pares e impares
    
    # Calcular el tiempo para este frame específico
    frame_position_ratio = frame_number / actual_frames_total
    
    # En lugar de sumar, restamos tiempo desde el tiempo final
    # para ir "hacia atrás" desde el timestamp final
    seconds_total = actual_frames_total / fps
    seconds_from_end = seconds_total * (1 - frame_position_ratio)
    
    # El timestamp para este frame es el timestamp base menos el tiempo desde el final
    current_datetime = base_datetime - timedelta(seconds=seconds_from_end)
    
    formatted_timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Incluir milisegundos
    return formatted_timestamp

def export_to_excel(output_path, video_name=""):
    """Exporta los datos de vehículos a un archivo Excel"""
    global vehicle_data
    
    if not vehicle_data:
        print("No hay datos de vehículos para exportar")
        return
    
    # Crear DataFrame con los datos recolectados
    df = pd.DataFrame(vehicle_data, columns=[
        'ID', 'Tipo de Vehículo', 'Trayectoria', 'Tiempo'
    ])
    
    # Obtener el nombre del archivo de video para nombrar el Excel
    file_prefix = ""
    if video_name:
        # Extraer la primera parte del nombre (antes del primer guion bajo)
        parts = os.path.basename(video_name).split('_')
        if parts:
            file_prefix = parts[0] + "_"
    
    # Crear un archivo Excel con múltiples hojas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(output_path, f'{file_prefix}deteccion_vehiculos_{timestamp}.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Datos_Detallados', index=False)
    
    print(f"Datos exportados a: {excel_path}")


class DetectionPredictor(BasePredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Contador de frames para procesar solo los impares
        self.frame_count = 0
        
        # Obtener FPS del video de entrada
        try:
            source_path = str(self.args.source)
            if os.path.isfile(source_path) and source_path.lower().endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(source_path)
                if cap.isOpened():
                    global fps
                    fps_value = cap.get(cv2.CAP_PROP_FPS)
                    if fps_value > 0:
                        fps = fps_value
                cap.release()
                
                # Extraer timestamp base del nombre del archivo
                extract_base_datetime(os.path.basename(source_path))
        except Exception:
            pass
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        global vehicle_data, frame_count
        
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
            
        # Incrementar el contador de frames
        self.frame_count += 1
        frame_count = self.frame_count  # Actualizar contador global
        
        # Procesar solo frames impares para mayor velocidad
        # Para frames pares, simplemente retornar sin procesar
        if self.frame_count % 2 == 0:
            return log_string
        
        # Tracker
        self.data_path = p
    
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
            
        # Filtrar solo las detecciones de vehículos (carros, motos, buses, etc.)
        vehicle_mask = torch.zeros_like(det[:, 5], dtype=torch.bool)
        for vehicle_class in VEHICLE_CLASSES:
            vehicle_mask = vehicle_mask | (det[:, 5] == vehicle_class)
            
        det = det[vehicle_mask]
        
        if len(det) == 0:
            return log_string
            
        # Contar detecciones por clase
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detecciones por clase
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    
        # Preparar detecciones para el tracker
        dets_to_sort = np.empty((0, 6))
        
        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x1, y1, x2, y2, conf, detclass])))
        
        # Actualizar el tracker
        tracked_dets = tracker.update(dets_to_sort)
        tracks = tracker.getTrackers()
        
        # Calcular timestamp basado en el número de frame
        timestamp = calculate_timestamp(self.frame_count)
        
        # Dibujar las trayectorias
        for track in tracks:
            # Guardar datos para Excel
            if hasattr(track, 'detclass') and len(track.centroidarr) > 1:
                vehicle_type = self.model.names[int(track.detclass)]
                
                # Crear string de la trayectoria
                trajectory = []
                for centroid in track.centroidarr:
                    trajectory.append(f"({int(centroid[0])},{int(centroid[1])})")
                trajectory_str = " -> ".join(trajectory)
                
                # Almacenar datos para Excel
                vehicle_data.append([
                    track.id,
                    vehicle_type,
                    trajectory_str,
                    timestamp
                ])
            
            # Dibujar líneas de trayectoria
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        rand_color_list[track.id], thickness=3) 
                        for i, _ in enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1]
        
        # Dibujar cajas delimitadoras si hay detecciones
        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:, :4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)
        
        # Mostrar información de clases en la pantalla
        for category in np.unique(tracked_dets[:, 4]) if len(tracked_dets) > 0 else []:
            count = np.sum(tracked_dets[:, 4] == category)
            category_name = self.model.names[int(category)]
            cv2.putText(im0, f"{category_name}: {count}", 
                       (20, 30 + 30 * int(category)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    random_color_list()
    
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    
    # Crear directorio para guardar resultados
    output_dir = os.path.join(ROOT, "resultados")
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener el nombre del video para nombrar el archivo Excel
    video_name = ""
    try:
        # Convertir cfg.source a string para manejar diferentes tipos de entrada
        source_path = str(cfg.source)
        if os.path.isfile(source_path):
            video_name = os.path.basename(source_path)
    except Exception:
        pass
    
    # Ejecutar la detección
    predictor = DetectionPredictor(cfg)
    predictor()
    
    # Exportar a Excel
    export_to_excel(output_dir, video_name)


if __name__ == "__main__":
    predict()
