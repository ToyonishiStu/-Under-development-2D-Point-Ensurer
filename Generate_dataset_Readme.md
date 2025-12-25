# KITTI to 2D LiDAR Point Completion Dataset

KITTIの3D LiDARデータセットを2D LiDAR点群補完タスク用のデータセットに変換するツール

## 概要

このプロジェクトは、KITTI Object Detection Datasetの3D LiDAR点群データ（`.bin`ファイル）を加工し、2D LiDAR点群補完（Point Completion）タスク用の教師ありデータセットを生成します。

生成方針は**PCN (Point Completion Network)** と同一思想：
- 完全点群 → 人工的欠損 → 補完

## 特徴

- **3D → 2D 変換**: KITTI 3D点群を水平面2D LiDARスキャンに変換
- **人工的欠損生成**: 連続角度領域の欠損（20-50%）
- **品質保証**: NaN/Inf チェック、部分集合検証など
- **PyTorch対応**: すぐに学習に使えるDatasetクラス付属
- **可視化ツール**: データセットの品質を視覚的に確認

## インストール

### 必要な環境

- Python 3.8+
- NumPy
- Matplotlib
- PyTorch（学習時のみ）

### 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. データセット生成

KITTIデータセットから2D LiDAR補完データセットを生成：

```bash
python generate_dataset.py \
  --kitti_root /path/to/KITTI \
  --output_root /path/to/output
```

#### オプション引数

- `--n_beams`: LiDARビーム数（デフォルト: 360）
- `--r_max`: 最大距離 [m]（デフォルト: 30.0）
- `--z_min`: 地面除去の閾値 [m]（デフォルト: -1.5）
- `--occlusion_min`: 最小欠損率（デフォルト: 0.2）
- `--occlusion_max`: 最大欠損率（デフォルト: 0.5）
- `--train_ratio`: 学習/検証の分割比率（デフォルト: 0.9）
- `--seed`: 乱数シード（デフォルト: 42）
- `--max_samples`: 処理する最大サンプル数（デフォルト: None = 全て）

#### 出力ディレクトリ構造

```
output/
├── train/
│   ├── 000000.npz
│   ├── 000001.npz
│   └── ...
├── val/
│   ├── 000100.npz
│   └── ...
└── dataset_stats.npz
```

### 2. データセット検証・可視化

生成されたデータセットの品質を確認：

```bash
# 検証のみ
python visualize_dataset.py \
  --dataset_root /path/to/output \
  --validate_only

# 可視化も実行（画像を保存）
python visualize_dataset.py \
  --dataset_root /path/to/output \
  --num_samples 10 \
  --output_dir ./visualizations
```

### 3. PyTorchでの使用

```python
from dataset import create_dataloaders

# データローダーを作成
train_loader, val_loader = create_dataloaders(
    dataset_root="/path/to/output",
    batch_size=32,
    num_workers=4
)

# 学習ループ
for partial, complete in train_loader:
    # partial: (batch_size, n_beams)
    # complete: (batch_size, n_beams)
    
    # モデルに入力
    output = model(partial)
    loss = criterion(output, complete)
    # ...
```

## データ仕様

### 入力データ

- **ソース**: KITTI Object Detection Dataset
- **ファイル形式**: `.bin` (float32配列)
- **座標系**: x=前方, y=左, z=上
- **データ**: `[x, y, z, reflectance]` × N点

### 出力データ

- **ファイル形式**: `.npz` (NumPy compressed)
- **含まれるデータ**:
  - `partial`: shape `(n_beams,)` - 欠損を含む入力スキャン
  - `complete`: shape `(n_beams,)` - 完全な出力スキャン
- **データ型**: float32
- **単位**: メートル

### 処理フロー

```
KITTI 3D点群 (.bin)
    ↓
前処理（距離制限・地面除去）
    ↓
2D LiDAR化（角度ビンごとに最小距離）
    ↓
complete（完全スキャン）
    ↓
欠損生成（連続角度領域を0に）
    ↓
partial（入力）
```

## 品質保証

生成されたデータセットは以下の条件を満たします：

- ✅ `partial` は `complete` の部分集合
- ✅ 欠損率がサンプルごとに異なる
- ✅ NaN / Inf が存在しない
- ✅ shape が全サンプルで一致
- ✅ 角度順が保持されている

## ファイル説明

- `generate_dataset.py`: データセット生成スクリプト
- `visualize_dataset.py`: 可視化・検証スクリプト
- `dataset.py`: PyTorch Dataset/DataLoaderクラス
- `requirements.txt`: 依存パッケージリスト

## 参考文献

- **PCN: Point Completion Network**
  - Paper: [arXiv:1808.00671](https://arxiv.org/abs/1808.00671)
  - Implementation: [qinglew/PCN-PyTorch](https://github.com/qinglew/PCN-PyTorch)
- **KITTI Vision Benchmark Suite**
  - Website: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)

## ライセンス

このコードは研究・教育目的で自由に使用できます。
KITTIデータセットの使用については、KITTIのライセンス条項に従ってください。

## トラブルシューティング

### Q: `.bin`ファイルが見つからない

A: `--kitti_root`が正しく設定されているか確認してください。
   必要なパス: `<kitti_root>/training/velodyne/*.bin`

### Q: 生成されるサンプル数が少ない

A: 以下の原因が考えられます：
- 点群が疎すぎる（`r_max`を大きくする）
- 地面除去が厳しすぎる（`z_min`を調整）
- 品質チェックで弾かれている（ログを確認）

### Q: メモリ不足エラー

A: `--max_samples`で処理するサンプル数を制限してください

## TODO

- [ ] より高度な地面除去アルゴリズム
- [ ] 複数種類の欠損パターン（ランダム点欠損など）
- [ ] データ拡張のバリエーション追加
- [ ] TensorBoard対応の可視化