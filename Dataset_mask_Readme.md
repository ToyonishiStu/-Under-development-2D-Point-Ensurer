# Dataset.py 改修版 使用ガイド

## 変更点の概要

`dataset.py` に **mask チャンネル** を追加しました。

### 改修前
```python
partial, target = dataset[0]
# partial: (360,) - 欠損あり
# target: (360,) - 完全データ
```

### 改修後
```python
partial, mask, target = dataset[0]
# partial: (360,) - 欠損あり
# mask: (360,) - 欠損マスク
# target: (360,) - 完全データ
```

---

## mask の仕様

### 定義
- **mask[i] = 1.0**: 観測あり（partial[i] > 0）
- **mask[i] = 0.0**: 欠損（partial[i] == 0）

### 生成方法
```python
mask = (partial > 0).astype(np.float32)
```

### 特徴
- **dtype**: `float32`
- **shape**: `(N,)` where N = ビーム数（例: 360）
- **正規化の対象外**: mask は常に `{0.0, 1.0}` の値

---

## 使用方法

### 1. 基本的な使い方

```python
from dataset import create_dataloaders

# データローダーを作成（既存のコードと同じ）
train_loader, val_loader = create_dataloaders(
    dataset_root="./output",
    batch_size=32,
    num_workers=4
)

# 学習ループ
for partial, mask, target in train_loader:
    # partial: (B, N) - 欠損あり入力
    # mask: (B, N) - 欠損マスク
    # target: (B, N) - 完全データ（教師信号）
    
    # モデルに入力
    output = model(partial, mask)  # maskをモデルに渡すことも可能
    
    # 損失計算（例: マスク部分のみで損失を計算）
    loss = criterion(output * mask, target * mask)
```

### 2. データセットクラスを直接使う

```python
from dataset import LiDAR2DCompletionDataset

dataset = LiDAR2DCompletionDataset(
    data_dir="./output/train",
    normalize=True,
    augment=True
)

# 1サンプル取得
partial, mask, target = dataset[0]

print(f"Partial: {partial.shape}")  # torch.Size([360])
print(f"Mask: {mask.shape}")        # torch.Size([360])
print(f"Target: {target.shape}")    # torch.Size([360])

# 欠損率を確認
occlusion_rate = (mask == 0).sum() / len(mask)
print(f"Occlusion rate: {occlusion_rate:.2%}")
```

### 3. バッチ処理での形状

```python
for partial, mask, target in train_loader:
    print(partial.shape)  # torch.Size([32, 360])
    print(mask.shape)     # torch.Size([32, 360])
    print(target.shape)   # torch.Size([32, 360])
    break
```

---

## モデル実装例

### 例1: maskを使わないシンプルなモデル

```python
import torch.nn as nn

class SimpleCompletionModel(nn.Module):
    def __init__(self, n_beams=360, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_beams, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_beams)
        )
    
    def forward(self, partial, mask=None):
        # maskは使わないが、インターフェースの互換性のため受け取る
        z = self.encoder(partial)
        output = self.decoder(z)
        return output

# 使用例
model = SimpleCompletionModel()
for partial, mask, target in train_loader:
    output = model(partial, mask)
    loss = criterion(output, target)
    # ...
```

### 例2: mask-aware なモデル

```python
class MaskAwareModel(nn.Module):
    def __init__(self, n_beams=360, hidden_dim=128):
        super().__init__()
        # partial と mask を結合して入力
        self.encoder = nn.Sequential(
            nn.Linear(n_beams * 2, hidden_dim),  # partial + mask
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_beams)
        )
    
    def forward(self, partial, mask):
        # partial と mask を結合
        x = torch.cat([partial, mask], dim=-1)
        z = self.encoder(x)
        output = self.decoder(z)
        return output

# 使用例
model = MaskAwareModel()
for partial, mask, target in train_loader:
    output = model(partial, mask)
    loss = criterion(output, target)
    # ...
```

### 例3: 欠損部分のみで損失を計算

```python
# マスクされた部分のみで損失を計算
for partial, mask, target in train_loader:
    output = model(partial, mask)
    
    # 欠損していた部分のみを評価
    occluded_mask = (mask == 0)
    
    # マスク適用
    loss = criterion(output[occluded_mask], target[occluded_mask])
    
    # または重み付き損失
    # loss = (criterion_per_point(output, target) * (1 - mask)).mean()
```

---

## テスト・検証方法

### 動作確認

```bash
python dataset.py --dataset_root ./output --batch_size 32
```

出力例:
```
Loaded 5000 samples from ./output/train
Computing normalization statistics...
  Distance normalization max: 28.45m
Loaded 555 samples from ./output/val

============================================================
DataLoader Test
============================================================

Batch shapes:
  Partial: torch.Size([32, 360])
  Mask: torch.Size([32, 360])
  Target: torch.Size([32, 360])

Value ranges:
  Partial: [0.000, 0.987]
  Mask: [0.000, 1.000]
  Target: [0.000, 1.000]

Mask statistics (first sample in batch):
  Observed points (mask=1): 234
  Occluded points (mask=0): 126
  Occlusion rate: 35.00%

Train batches: 157
Val batches: 18
```

### mask の妥当性チェック

```python
import numpy as np

# データセットから1サンプル取得
partial, mask, target = dataset[0]

# 1. maskが0/1のみであることを確認
assert torch.all((mask == 0) | (mask == 1))

# 2. partial > 0 の位置でmask = 1 であることを確認
assert torch.all(mask[partial > 0] == 1)

# 3. partial == 0 の位置でmask = 0 であることを確認
assert torch.all(mask[partial == 0] == 0)

print("✓ Mask validation passed!")
```

---

## 後方互換性について

既存のコードで `partial, complete = dataset[0]` のように2要素で受け取っている場合、以下のエラーが発生します：

```python
ValueError: too many values to unpack (expected 2)
```

**対処法**:
```python
# 修正前
partial, complete = dataset[0]

# 修正後（maskを受け取る）
partial, mask, target = dataset[0]

# または、maskを使わない場合
partial, _, target = dataset[0]
```

---

## まとめ

### 主な変更点
- `__getitem__` の返り値: `(partial, complete)` → `(partial, mask, target)`
- `mask` チャンネルの追加
- complete → target への名前変更（意味的に明確化）

### メリット
- mask-aware なモデルの実装が容易
- 欠損パターンの明示的な表現
- 将来の拡張性向上（異なる欠損定義への対応）

### 互換性
- 既存の `.npz` ファイルはそのまま使用可能
- Dataset の初期化引数は変更なし
- `create_dataloaders` の使い方も変更なし