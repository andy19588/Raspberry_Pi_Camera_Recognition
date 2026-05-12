import cv2
import numpy as np
import argparse
import threading
import time

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

    print(f"⏳ 載入 {model_type.upper()} 模型...")
    if model_type == 'svm':
        import joblib
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return
    else:
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return
        
    print("✅ 模型載入完成！正在開啟攝影機...")

    cap = cv2.VideoCapture(0)
    # Raspberry Pi 攝影機設定，降低解析度以提高 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    labels = ['Rock', 'Paper', 'Scissors']
    pred_label = "Waiting..."
    
    # 用於多執行緒共享的變數
    current_roi = None
    lock = threading.Lock()

    def predict_thread():
        nonlocal pred_label, current_roi
        while True:
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
                        pred_idx = model.predict(features)[0]
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
                        pred_idx = np.argmax(preds[0])
                        pred_label = labels[pred_idx]
                except Exception as e:
                    pass
            time.sleep(0.01) # 讓出 CPU 資源避免吃滿

    # 啟動背景辨識執行緒
    t = threading.Thread(target=predict_thread, daemon=True)
    t.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 無法讀取攝影機畫面")
            break
            
        # 繪製讓使用者放置手部的辨識框
        h, w = frame.shape[:2]
        roi_size = 224
        x1, y1 = w//2 - roi_size//2, h//2 - roi_size//2
        x2, y2 = w//2 + roi_size//2, h//2 + roi_size//2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 擷取 ROI 區域並交給背景執行緒
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            with lock:
                current_roi = roi

        cv2.putText(frame, f"Predict: {pred_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Raspberry Pi - Gesture Recognition", frame)

        # 按 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
