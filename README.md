
---
## はじめに

監視カメラやロボットの視覚認識など、2D画像から**人の3次元的な位置**（奥行きや座標）を推定したい場面は多くあります。

本記事では、以下の2つの技術を組み合わせて、**人物のカメラ座標（X, Y, Z）を推定**するシステムを実装してみます。

* **YOLOv8**：物体検出（人物検出）
* **MiDaS**：単眼カメラでの深度（距離）推定

---

## システムの概要

本システムでは、次のステップで画像内の人物の位置を推定します：

1. カメラ画像を読み込む
2. MiDaSで画像の深度マップ（距離情報）を推定
3. YOLOv8で人物を検出
4. 検出した人物のバウンディングボックス内の深度を取得し、カメラ座標（X, Y, Z）を推定
5. 結果を可視化（画像とコンソールに出力）


---

## 使用したライブラリ・環境

* Python 3.9+
* OpenCV
* torch（PyTorch）
* ultralytics（YOLOv8用）
* intel-isl/MiDaS（深度推定）

事前に以下をインストールしておきます：

```bash
pip install opencv-python torch torchvision torchaudio
pip install ultralytics
```

---

## コード解説

### 1. モデルと画像の読み込み

```python
# 画像読み込み
frame = cv2.imread(image_path)

# MiDaSモデル読み込み
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# YOLOv8モデル読み込み
model = YOLO("yolov8n.pt")
```

ここでは、MiDaS（軽量版）とYOLOv8（Nano）モデルを読み込みます。

---

### 2. 深度マップの推定

```python
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_tensor = midas_transforms(img_rgb)

with torch.no_grad():
    prediction = midas(input_tensor)
    depth_map = prediction.squeeze().cpu().numpy()
```

MiDaSを使って画像全体の相対的な深度（距離）を推定します。

---

### 3. YOLOで人物検出し、深度から3D座標へ

```python
results = model(frame)[0]

for box in results.boxes:
    if int(box.cls[0]) == 0:  # クラス0 = person
        ...
        z_rel = np.median(depth_roi)
        z = z_rel * scale_factor  # メートル換算
        Xc = (px - cx) * z / fx
        Yc = (py - cy) * z / fy
```

* バウンディングボックス中心を使ってカメラ座標に変換
* `scale_factor`は、深度マップをメートル換算するためのスケールです（※キャリブレーションにより調整が必要）

---

### 4. 結果の表示

```python
cv2.putText(annotated, f"XYZ: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f}) m", ...)
cv2.imshow("YOLO + Depth + XYZ", annotated)
```

推定したX, Y, Z座標を画像に描画し、表示します。

---

## 実行結果の例（ログ出力）

```
Person bbox: (110, 120)-(230, 360)
Depth (Z): 2.423 m
Camera Coords (X, Y, Z): (-0.582, -1.126, 2.423)
```

このように、**人物がカメラからどのくらいの距離にいるのか**、**左右・上下にどれだけ位置しているのか**がわかります。

---

## まとめ・応用例

このシステムを活用することで、以下のような応用が可能です：

* ドローンやロボットによる人追跡
* スマート監視カメラによる人物の距離把握
* 人と物体の位置関係の可視化（ARやXRにも応用可能）

### 注意点

* MiDaSは相対深度しかわからないため、**正確な距離推定にはカメラキャリブレーションが必要**です
* 処理速度や精度を求める場合はGPUでの実行を推奨します

---

## ソースコード全文

```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 画像のパスを指定
image_path = "/Users/smri/Downloads/VS code/b4/IPcamera_ws/snapshots/snapshot_20250526_131630.jpg"

# 画像読み込み
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"画像が見つかりません: {image_path}")

# MiDaSで深度推定モデル読み込み
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu").eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# YOLOv8モデル読み込み（人物検出）
model = YOLO("yolov8n.pt")

# --- キャリブレーション仮定値 ---
scale_factor = 3.0  # 相対深度をメートル換算

# 1. 深度マップ推定
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_tensor = midas_transforms(img_rgb)
with torch.no_grad():
    prediction = midas(input_tensor)
    depth_map = prediction.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

# 深度マップの正規化
depth_min, depth_max = depth_map.min(), depth_map.max()
depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

# 仮のカメラ内部パラメータ
fx = 500.0
fy = 500.0
cx = frame.shape[1] / 2
cy = frame.shape[0] / 2

# 2. 人物検出
results = model(frame)[0]
annotated = frame.copy()

for box in results.boxes:
    cls_id = int(box.cls[0])
    if cls_id == 0:  # person
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # バウンディングボックス内の深度の中央値
        depth_roi = depth_map_norm[y1:y2, x1:x2]
        if depth_roi.size == 0:
            continue
        z_rel = np.median(depth_roi)
        z = z_rel * scale_factor  # メートルに変換

        # ピクセル座標 → カメラ座標
        px, py = (x1 + x2) // 2, (y1 + y2) // 2
        Xc = (px - cx) * z / fx
        Yc = (py - cy) * z / fy
        Zc = z

        # 表示処理
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"XYZ: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f}) m", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # コンソール出力
        print(f"Person bbox: ({x1},{y1})-({x2},{y2})")
        print(f"Depth (Z): {Zc:.3f} m")
        print(f"Camera Coords (X, Y, Z): ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})\n")

# 結果表示
cv2.imshow("YOLO + Depth + XYZ", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

## 最後に

MiDaSとYOLOv8を組み合わせることで、単眼カメラだけでも人の3D位置推定が実現できることがわかりました。精度向上やリアルタイム化、ZKPによるプライバシー保護など、さらなる展開も期待できます。

興味がある方はぜひ試してみてください！

---

必要に応じて、記事用にソースコード全文も整形して載せることができます。もしQiitaやZenn用のマークダウン形式が必要であれば、整形して出力します。
