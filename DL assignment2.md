---
title: DL assignment2

---

# DL Assigment 2

## 1. 資料集
### 類別
* Imagenette-160：10類圖像分類，來源自ImageNet子集。
* Mini-VOC-Seg：PASCAL VOC 2012語意分割子集，21類（含背景），mask為PNG格式。
* Mini-COCO-Det：COCO 2017子集，選10類物件檢測，標註為COCO JSON格式。

### 數量
* Imagenette-160：train/val (240/60) ，從10個class中各抽(24/6)張
* Mini-VOC-Seg：隨機抽出300張，並將之分成train/val (240/60)
* Mini-COCO-Det：選10個類別('person', 'car', 'bicycle', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'dog')，每個類別抽(24/6)張，組成train/val (240/60)。



## 2. 模型設計
* **Backbone**：MobileNetV3-Small (預訓練)。
* **單一頭部**（MultiTaskHead）：
* **共享特徵抽取**：兩層卷積+BN+ReLU。
* **分割分支**：上採樣（bilinear, scale=32）+ 1x1 conv，輸出21通道，對應512x512分割圖。
* **檢測分支**：1x1 conv，輸出 6 + num_det_classes + num_cls_classes 通道（6為檢測參數，含座標、置信度、objectness）。
* **分類分支**：全局平均池化後接全連接層，輸出10類。

總參數量：3.15M < 8M，推理速度符合作業要求。

## 3. 訓練流程
### Stage 1：語意分割任務

使用 Mini-VOC-Seg 訓練分割分支。

計算 mIoU 作為基線。

### Stage 2：物件檢測任務

使用 Mini-COCO-Det 訓練檢測分支。

計算 mAP 作為基線。

### Stage 3：影像分類任務

使用 Imagenette-160 訓練分類分支。

計算 Top-1/Top-5 Accuracy 作為基線。

### EWC 防遺忘機制

每個任務訓練結束後，計算 Fisher 信息矩陣，保護已學任務的重要參數。
後續任務訓練時，損失函數加入 EWC 正則項。

## 4. EWC 實作細節
Fisher 信息矩陣計算時機：每個任務訓練結束後，遍歷該任務數據計算。

多任務Fisher累積：每次切換任務時，累積所有已學任務的Fisher資訊。

正則化項：對所有重要參數施加懲罰，防止偏離已學任務的最優解。

## 5. 主要程式架構
(1) 數據集載入與處理
分割：影像與mask同步resize，mask轉long tensor，忽略255。

檢測：COCO格式解析，bbox轉[x1,y1,x2,y2]，類別ID映射，目標資料為字典列表。

分類：影像resize+normalize，label轉long tensor。

(2) 訓練與評估
每個任務獨立訓練，評估指標分別為 mIoU、mAP、Top-1/Top-5 Acc。

訓練循環中，根據任務動態選擇損失函數與資料處理方式。

(3) EWC 計算與整合
訓練完每個任務後，調用 set_ewc 計算並保存 Fisher 信息。

訓練新任務時，損失函數自動加上 EWC penalty。

## 6. 主要挑戰與解決方案
多任務數據格式不一致：自訂 collate_fn，分任務處理目標資料結構。

標籤越界與設備錯誤：每步都檢查標籤範圍，所有張量明確轉移到正確device。

EWC計算效率：僅在每個任務訓練結束後計算 Fisher，避免重複計算。

評估流程穩健：所有評估指標計算前，將張量轉到CPU並轉為numpy。