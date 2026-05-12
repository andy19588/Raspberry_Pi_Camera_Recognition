import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(folder_path, img_size=(224, 224)):
    images = []
    labels = []
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        category_path = os.path.join(folder_path, category)
        
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path):
            print(f"⚠️ 找不到 {category} 的資料夾 -> {category_path}")
            continue
            
        print(f"📂 正在載入 {category} 的圖片...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(img, img_size)
                    images.append(resized)
                    labels.append(label_idx)
                    
    return np.array(images), np.array(labels)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== 步驟 1: 讀取圖片 (EfficientNet) ===")
    X_train, y_train = load_images_from_folder(train_dir)
    X_test, y_test = load_images_from_folder(test_dir)

    if len(X_train) == 0:
        print("❌ 找不到訓練圖片！")
        return

    # 前處理
    X_train = tf.keras.applications.efficientnet.preprocess_input(X_train.astype(np.float32))
    X_test = tf.keras.applications.efficientnet.preprocess_input(X_test.astype(np.float32))
    
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    print("\n=== 步驟 2: 建立與訓練 EfficientNet 模型 ===")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train_cat, epochs=5, batch_size=32, validation_data=(X_test, y_test_cat))

    print("\n=== 步驟 3: 評估與儲存模型 ===")
    loss, accuracy = model.evaluate(X_test, y_test_cat)
    print(f"🎯 測試集準確率: {accuracy * 100:.2f}%")

    os.makedirs(demo_dir, exist_ok=True)
    model_path = os.path.join(demo_dir, 'rps_efficientnet_model.h5')
    model.save(model_path)
    print(f"✅ 模型已儲存於: {model_path}")

if __name__ == "__main__":
    main()
