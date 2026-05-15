import cv2
import numpy as np
import argparse
import threading
import time
import mediapipe as mp
from PIL import Image, ImageFont, ImageDraw
import platform

def put_chinese_text(img, text, position, text_color=(0, 255, 0), font_size=40):
    if not isinstance(img, np.ndarray):
        return img
    
    font_path = None
    system = platform.system()
    if system == "Windows":
        font_paths = ["C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/simhei.ttf"]
    elif system == "Linux":
        font_paths = ["/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 
                      "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
    else:
        font_paths = []
        
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except IOError:
            continue
            
    if font is None:
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Gesture Recognition")
    parser.add_argument('--model', type=str, default='mobilenet', choices=['svm', 'mobilenet', 'efficientnet'], help="Model to use (svm, mobilenet, efficientnet)")
    args = parser.parse_args()

    model_type = args.model
    if model_type == 'svm':
        model_path = 'rps_svm_model.pkl'
    elif model_type == 'mobilenet':
        model_path = 'rps_mobilenet_model.h5'
    else:
        model_path = 'rps_efficientnet_model.h5'

    print(f"載入 {model_type.upper()} 模型...")
    if model_type == 'svm':
        import joblib
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"載入失敗: {e}")
            return
    else:
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"載入失敗: {e}")
            return
        
    print("模型載入完成！正在開啟攝影機...")

    cap = cv2.VideoCapture(0)
    # Raspberry Pi 攝影機設定，降低解析度以提高 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    labels = ['石頭', '布', '剪刀']
    pred_label = "等待中..."
    
    # 用於多執行緒共享的變數
    current_roi = None
    lock = threading.Lock()
    new_frame_event = threading.Event()

    def predict_thread():
        nonlocal pred_label, current_roi
        while True:
            new_frame_event.wait() # 等待新畫面才進行推論
            new_frame_event.clear()
            
            roi_copy = None
            with lock:
                if current_roi is not None:
                    roi_copy = current_roi.copy()
            
            if roi_copy is not None:
                try:
                    if model_type == 'svm':
                        gray = cv2.cvtColor(roi_copy, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (64, 64))
                        features = (resized.flatten() / 255.0).reshape(1, -1)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(features)[0]
                            max_prob = float(np.max(probs))
                            if max_prob > 0.7:  # SVM 信心度門檻
                                pred_idx = int(np.argmax(probs))
                                pred_label = labels[pred_idx]
                            else:
                                pred_label = "其他手勢"
                        else:
                            # 若 SVM 沒開啟 probability=True，只能直接給結果
                            pred_idx = int(model.predict(features)[0])
                            pred_label = labels[pred_idx]
                    else:
                        rgb = cv2.cvtColor(roi_copy, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(rgb, (224, 224))
                        input_arr = np.expand_dims(resized, axis=0).astype(np.float32)
                        
                        if model_type == 'mobilenet':
                            input_arr = tf.keras.applications.mobilenet_v2.preprocess_input(input_arr)
                        else:
                            input_arr = tf.keras.applications.efficientnet.preprocess_input(input_arr)
                            
                        preds = model.predict(input_arr, verbose=0)
                        max_prob = float(np.max(preds[0]))
                        if max_prob > 0.7:  # 深度學習模型信心度門檻
                            pred_idx = int(np.argmax(preds[0]))
                            pred_label = f"{labels[pred_idx]} ({max_prob*100:.0f}%)"
                        else:
                            pred_label = "其他手勢"
                except Exception as e:
                    pass

    # 啟動背景辨識執行緒
    t = threading.Thread(target=predict_thread, daemon=True)
    t.start()

    # 初始化 MediaPipe Tasks Hands (支援 Python 3.13)
    import os
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    task_path = 'hand_landmarker.task'
    if not os.path.exists(task_path):
        print("正在下載 MediaPipe 模型檔案 (hand_landmarker.task)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_path
        )
        print("下載完成！")

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    mp_hands_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break
            
        # 使用 MediaPipe 偵測手部
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                h, w, _ = frame.shape
                
                # 手動繪製骨架與節點
                for p1, p2 in mp_hands_connections:
                    x1, y1 = int(hand_landmarks[p1].x * w), int(hand_landmarks[p1].y * h)
                    x2, y2 = int(hand_landmarks[p2].x * w), int(hand_landmarks[p2].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

                # 取得手部邊界框
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # 增加 padding 讓手部完整進入 ROI
                padding = 40
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # 擷取 ROI 區域並觸發背景執行緒
                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size > 0:
                    with lock:
                        current_roi = roi
                    new_frame_event.set() # 喚醒推論執行緒
                break # 確保只處理第一隻手
        else:
            pred_label = "其他"

        # 使用 PIL 畫中文
        frame = put_chinese_text(frame, f"預測結果: {pred_label}", (10, 10), (0, 0, 255), 40)
        
        cv2.imshow("Raspberry Pi - Gesture Recognition", frame)

        # 按 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
