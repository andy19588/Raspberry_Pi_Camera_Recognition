# Raspberry Pi Hand Gesture Recognition

此專案為在樹莓派上執行的手勢辨識 (剪刀、石頭、布) 實作，並包含了三種不同模型的訓練與測試程式。

## 目錄結構
- `dataset/`: 存放訓練 (train) 與測試 (test) 的影像資料集。
- `train/`: 包含訓練各種模型的程式碼與相依套件列表。
  - `train_svm.py`: 訓練 SVM 模型
  - `train_mobilenet.py`: 訓練 MobileNetV2 模型
  - `train_efficientnet.py`: 訓練 EfficientNetB0 模型
  - `requirements.txt`: 訓練環境的套件需求
- `demo/`: 存放訓練好的模型及實際應用於測試與相機即時辨識的程式。
  - `test.py`: 測試模型的準確率與分類報告
  - `pi_camera_recognition.py`: 樹莓派上使用的即時攝影機辨識程式
  - `carema.py`: 基本攝影機測試
  - `requirements.txt`: 執行測試與辨識環境的套件需求

## 模型訓練

若要重新訓練模型，請進入 `train` 目錄並執行對應的訓練程式：
```bash
cd train
pip install -r requirements.txt
# 訓練 SVM 模型
python train_svm.py
# 訓練 MobileNet 模型
python train_mobilenet.py
# 訓練 EfficientNet 模型
python train_efficientnet.py
```
訓練好的模型會自動儲存至 `demo/` 資料夾中。

## 模型測試

可以使用 `demo/test.py` 來測試模型在 `dataset/test` 測試集上的表現 (包含 Accuracy, Precision, Recall, F1-score 等)：
```bash
cd demo
# 測試 SVM
python test.py --model svm
# 測試 MobileNet
python test.py --model mobilenet
# 測試 EfficientNet
python test.py --model efficientnet
```

## 樹莓派即時辨識

在樹莓派上啟動即時手勢辨識：
```bash
cd demo
pip install -r requirements.txt
```

執行即時辨識程式並指定使用的模型：
```bash
# 使用 SVM 辨識
python pi_camera_recognition.py --model svm
# 使用 MobileNet 辨識
python pi_camera_recognition.py --model mobilenet
# 使用 EfficientNet 辨識
python pi_camera_recognition.py --model efficientnet
```
執行後畫面中會出現一個綠色方框，請將手勢置於框內即可看到即時的辨識結果。按 `q` 鍵可離開程式。

---

## 樹梅派環境建立與刷機紀錄
1. 下載 [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. 選擇 Raspberry Pi 4 64-bit 系統並寫入 SD 卡。
3. 設定 WiFi、Hostname、SSH 以及對應時區。
4. 將 SD 卡插入樹莓派後開機，並透過 SSH 連入安裝上述需求套件。
