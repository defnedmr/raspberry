import os
import sys
import argparse
import glob
import time
import threading
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record',
                    help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Model yükleme (ana iş parçacığında bir kez yapılır)
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)
model = YOLO(model_path, task='detect')
labels = model.names

# Görüntü kaynağı belirleme
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']
source_type = None
cap = None
picam2 = None
imgs_list = None
usb_idx = None
picam_idx = None

if os.path.isdir(img_source):
    source_type = 'folder'
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1] in img_ext_list]
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
        imgs_list = [img_source]
    elif ext in vid_ext_list:
        source_type = 'video'
        cap = cv2.VideoCapture(img_source)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
    cap = cv2.VideoCapture(usb_idx)
elif 'picamera' in img_source:
    from picamera2 import Picamera2
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}) # Başlangıç çözünürlüğü
    if user_res:
        resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
        config = picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)})
    picam2.configure(config)
    picam2.start()
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Çözünürlük ayarlama
resW, resH = None, None
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    if cap and source_type in ['video', 'usb']:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# Kayıt ayarlama
recorder = None
if record:
    if source_type not in ['video', 'usb', 'picamera']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# İş parçacıkları için veri kuyrukları
frame_queue = deque(maxlen=1)
result_queue = deque(maxlen=1)
stop_event = threading.Event()

def read_frame():
    global img_count, frame_queue, stop_event, imgs_list, cap, picam2, resize, resW, resH, source_type

    if stop_event.is_set():
        return None

    frame = None
    if source_type == 'image' or source_type == 'folder':
        if img_count < len(imgs_list):
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
            if resize and frame is not None:
                frame = cv2.resize(frame, (resW, resH))
        else:
            stop_event.set()
    elif source_type == 'video' or source_type == 'usb':
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
                frame = None
            elif resize and frame is not None:
                frame = cv2.resize(frame, (resW, resH))
        else:
            stop_event.set()
    elif source_type == 'picamera':
        if picam2 and picam2.is_open:
            frame_bgra = picam2.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if resize and frame is not None:
                frame = cv2.resize(frame, (resW, resH))
        else:
            stop_event.set()

    return frame

def frame_reader_worker(frame_queue, stop_event, source_type, cap, picam2, imgs_list, resize, resW, resH):
    global img_count
    img_count = 0
    while not stop_event.is_set():
        frame = read_frame()
        if frame is not None:
            if not frame_queue:
                frame_queue.append(frame)
        else:
            time.sleep(0.01)
            if stop_event.is_set() and not frame_queue: # Son kareyi de işleme için
                break
            if source_type in ['image', 'folder'] and img_count >= len(imgs_list):
                stop_event.set()
                break
            elif source_type in ['video', 'usb'] and (cap is None or not cap.isOpened()):
                stop_event.set()
                break
            elif source_type == 'picamera' and (picam2 is None or not picam2.is_open):
                stop_event.set()
                break
            elif frame is None and source_type not in ['image', 'folder']: # Video/kamera sonu
                stop_event.set()
                break
            elif frame is None and source_type in ['image', 'folder'] and img_count >= len(imgs_list):
                stop_event.set()
                break
            elif frame is None:
                time.sleep(0.01)

def perform_inference(frame, model):
    if frame is not None:
        results = model(frame, verbose=False)
        return frame, results
    return None, None

def inference_worker(frame_queue, result_queue, stop_event, model):
    while not stop_event.is_set():
        if frame_queue:
            frame = frame_queue.popleft()
            frame_with_results = perform_inference(frame, model)
            if frame_with_results[0] is not None:
                if not result_queue:
                    result_queue.append(frame_with_results)
        else:
            time.sleep(0.01)

def process_results_and_draw(frame, results, labels, min_thresh, bbox_colors=None):
    if frame is None or results is None:
        return None
    detections = results[0].boxes
    object_count = 0

    if bbox_colors is None:
        bbox_colors = [(0, 255, 0)] # Varsayılan yeşil renk

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1) # Orta nokta çizimi
            label = f'{classname}: {conf:.2f}'
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            object_count += 1
    return frame

def calculate_fps(frame_rate_buffer, fps_avg_len):
    if frame_rate_buffer:
        return np.mean(frame_rate_buffer)
    return 0.0

def init_recorder(record, source_type, user_res, record_name='demo1.avi', record_fps=30):
    if record:
        if source_type not in ['video', 'usb', 'picamera']:
            print('Recording only works for video and camera sources.')
            return None
        if not user_res:
            print('Please specify resolution to record video at.')
            return None
        resW, resH = map(int, user_res.split('x'))
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))
        return recorder
    return None

def record_frame(recorder, frame):
    if recorder is not None and frame is not None:
        recorder.write(frame)

def release_recorder(recorder):
    if recorder is not None:
        recorder.release()

if __name__ == "__main__":
    # Argüman ayrıştırma ve global değişkenlerin tanımlanması (aynı kalır)

    # Model yükleme (aynı kalır)

    # Görüntü kaynağı başlatma (kaynak tipine göre)
    # ...

    # Çözünürlük ayarlama (aynı kalır)

    # Kayıt başlatma
    recorder = init_recorder(record, source_type, user_res)

    # İş parçacıkları için veri kuyrukları ve durdurma eventi
    frame_queue = deque(maxlen=1)
    result_queue = deque(maxlen=1)
    stop_event = threading.Event()

    # İş parçacıklarını başlatma
    reader_thread = None
    if source_type in ['image', 'folder']:
        reader_thread = threading.Thread(target=frame_reader_worker, args=(frame_queue, stop_event, source_type, cap, picam2, imgs_list, resize, resW, resH))
    elif source_type in ['video', 'usb']:
        reader_thread = threading.Thread(target=frame_reader_worker, args=(frame_queue, stop_event, source_type, cap, picam2, imgs_list, resize, resW, resH))
    elif source_type == 'picamera':
        reader_thread = threading.Thread(target=frame_reader_worker, args=(frame_queue, stop_event, source_type, cap, picam2, imgs_list, resize, resW, resH))

    inference_thread = threading.Thread(target=inference_worker, args=(frame_queue, result_queue, stop_event, model))

    if reader_thread:
        reader_thread.start()
    inference_thread.start()

    # Ana döngü (görüntüleme ve kayıt)
    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    last_time = time.perf_counter()

    while not stop_event.is_set():
        if result_queue:
            frame, results = result_queue.popleft()
            processed_frame = process_results_and_draw(frame, results, labels, min_thresh)

            if processed_frame is not None:
                current_time = time.perf_counter()
                frame_time = current_time - last_time
                if frame_time > 0:  # Bölme sıfır hatasını önle
                    fps = 1 / frame_time
                    frame_rate_buffer.append(fps)
                    if len(frame_rate_buffer) > fps_avg_len:
                        frame_rate_buffer.pop(0)
                    avg_frame_rate = calculate_fps(frame_rate_buffer, fps_avg_len)
                    cv2.putText(processed_frame, f'FPS: {avg_frame_rate:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)
                cv2.imshow('YOLO Detection', processed_frame)
                record_frame(recorder, processed_frame)
                last_time = current_time  # Sadece kare işlendiyse güncelle
        else:
            time.sleep(0.001)  # Kısa bir süre bekle, CPU kullanımını azalt
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop_event.set()
        elif key == ord('s'):
            cv2.waitKey(0)

    # İş parçacıklarını bekleme ve temizleme
    if reader_thread:
        reader_thread.join()
    inference_thread.join()

    # Kayıtçı serbest bırakma
    release_recorder(recorder)

    if cap is not None:
        cap.release()
    if picam2 is not None:
        picam2.stop()
        picam2.close()
    cv2.destroyAllWindows()
