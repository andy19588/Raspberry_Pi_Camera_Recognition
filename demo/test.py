import os
import cv2
import numpy as np
import joblib
import argparse
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser(description="Test RPS Models")
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'mobilenet', 'efficientnet'], help="Model to test (svm, mobilenet, efficientnet)")
    args = parser.parse_args()

    model_type = args.model
    if model_type == 'svm':
        model_path = 'rps_svm_model.pkl'
    elif model_type == 'mobilenet':
        model_path = 'rps_mobilenet_model.h5'
    else:
        model_path = 'rps_efficientnet_model.h5'
        
    test_dir = '../dataset/test' 

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'，請確認是否已放入 demo 資料夾。")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ 錯誤：找不到測試資料集 '{test_dir}'。")
        print("請確認 dataset/test 資料夾是否存在於上一層目錄。")
        return

    # 載入模型
    print(f"⏳ 載入 {model_type.upper()} 模型中...")
    if model_type == 'svm':
        clf = joblib.load(model_path)
    else:
        import tensorflow as tf
        clf = tf.keras.models.load_model(model_path)
    print("✅ 模型載入成功！\n")

    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    X_test, y_test = [], []

    # 讀取並處理測試圖片
    print("📂 正在讀取測試集圖片並進行預測...")
    for category, label_idx in label_map.items():
        category_path = os.path.join(test_dir, category)
        
        # 處理資料夾內可能多包一層的情況
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                category_path = os.path.join(test_dir, subdirs[0], category)

        if not os.path.exists(category_path):
            print(f"⚠️ 找不到 {category} 的圖片資料夾，略過...")
            continue
            
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 資料前處理
                    if model_type == 'svm':
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (64, 64))
                        X_test.append(resized.flatten())
                    else:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(img_rgb, (224, 224))
                        X_test.append(resized)
                    y_test.append(label_idx)

    if not X_test:
        print("❌ 錯誤：沒有讀取到任何圖片，請檢查資料夾結構。")
        return

    # 正規化與預測
    if model_type == 'svm':
        X_test = np.array(X_test) / 255.0
        y_test = np.array(y_test)
        y_pred = clf.predict(X_test)
    else:
        X_test = np.array(X_test).astype(np.float32)
        if model_type == 'mobilenet':
            X_test = tf.keras.applications.mobilenet_v2.preprocess_input(X_test)
        else:
            X_test = tf.keras.applications.efficientnet.preprocess_input(X_test)
        
        y_test = np.array(y_test)
        y_pred_probs = clf.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 {model_type.upper()} 測試結果統整:")
    print(f"總共測試了 {len(y_test)} 張圖片")
    print(f"🎯 模型準確率: {accuracy * 100:.2f}%\n")
    print("📝 分類詳細報告:")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

if __name__ == "__main__":
    main()