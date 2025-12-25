#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for 2D LiDAR Point Completion

PyTorchでの学習に使用するデータセットクラス
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class LiDAR2DCompletionDataset(Dataset):
    """2D LiDAR点群補完用データセット"""
    
    def __init__(
        self,
        data_dir: str,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Args:
            data_dir: データディレクトリ（train or val）
            normalize: 距離を正規化するか
            augment: データ拡張を行うか（回転など）
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.augment = augment
        
        # .npz ファイルのリストを取得
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")
        
        print(f"Loaded {len(self.file_list)} samples from {data_dir}")
        
        # 正規化のための統計量を計算
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """正規化のための統計量を計算"""
        print("Computing normalization statistics...")
        
        max_distances = []
        
        # サンプル数が多い場合は一部のみ使用
        sample_size = min(1000, len(self.file_list))
        indices = np.random.choice(len(self.file_list), sample_size, replace=False)
        
        for idx in indices:
            data = np.load(self.file_list[idx])
            complete = data['complete']
            # 0より大きい値の最大値を記録
            valid_distances = complete[complete > 0]
            if len(valid_distances) > 0:
                max_distances.append(np.max(valid_distances))
        
        self.distance_max = np.percentile(max_distances, 95)  # 95パーセンタイルを使用
        print(f"  Distance normalization max: {self.distance_max:.2f}m")
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            partial: shape (n_beams,)
            complete: shape (n_beams,)
        """
        # データを読み込み
        data = np.load(self.file_list[idx])
        partial = data['partial'].astype(np.float32)
        complete = data['complete'].astype(np.float32)
        
        # データ拡張（回転）
        if self.augment:
            shift = np.random.randint(0, len(partial))
            partial = np.roll(partial, shift)
            complete = np.roll(complete, shift)
        
        # 正規化
        if self.normalize:
            partial = partial / self.distance_max
            complete = complete / self.distance_max
        
        # Tensorに変換
        partial = torch.from_numpy(partial)
        complete = torch.from_numpy(complete)
        
        return partial, complete


def create_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを作成
    
    Args:
        dataset_root: データセットのルートディレクトリ
        batch_size: バッチサイズ
        num_workers: データローディングのワーカー数
        normalize: 距離を正規化するか
        augment_train: 学習データに拡張を適用するか
        
    Returns:
        train_loader, val_loader
    """
    # データセットを作成
    train_dataset = LiDAR2DCompletionDataset(
        data_dir=os.path.join(dataset_root, "train"),
        normalize=normalize,
        augment=augment_train
    )
    
    val_dataset = LiDAR2DCompletionDataset(
        data_dir=os.path.join(dataset_root, "val"),
        normalize=normalize,
        augment=False  # 検証データには拡張を適用しない
    )
    
    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# テスト用のスクリプト
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    # データローダーを作成
    train_loader, val_loader = create_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # サンプルを取得
    print("\n" + "=" * 60)
    print("DataLoader Test")
    print("=" * 60)
    
    for partial, complete in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Partial: {partial.shape}")
        print(f"  Complete: {complete.shape}")
        print(f"  Partial range: [{partial.min():.3f}, {partial.max():.3f}]")
        print(f"  Complete range: [{complete.min():.3f}, {complete.max():.3f}]")
        
        # 1バッチのみテスト
        break
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")