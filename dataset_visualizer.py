#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D LiDAR Point Completion Dataset Visualization and Validation

生成されたデータセットを可視化・検証するスクリプト
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from typing import Optional


class DatasetVisualizer:
    """データセットの可視化・検証クラス"""
    
    def __init__(self, dataset_root: str):
        """
        Args:
            dataset_root: データセットのルートディレクトリ
        """
        self.dataset_root = dataset_root
        self.train_dir = os.path.join(dataset_root, "train")
        self.val_dir = os.path.join(dataset_root, "val")
    
    def load_sample(self, npz_path: str):
        """
        サンプルを読み込む
        
        Args:
            npz_path: .npzファイルのパス
            
        Returns:
            partial, complete
        """
        data = np.load(npz_path)
        return data['partial'], data['complete']
    
    def validate_dataset(self):
        """
        データセット全体の品質を検証
        """
        print("=" * 60)
        print("Dataset Validation")
        print("=" * 60)
        
        # 統計情報を読み込み
        stats_path = os.path.join(self.dataset_root, "dataset_stats.npz")
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            print("\n[Dataset Statistics]")
            for key in stats.keys():
                print(f"  {key}: {stats[key]}")
        
        # train/val のサンプル数を確認
        train_files = glob.glob(os.path.join(self.train_dir, "*.npz"))
        val_files = glob.glob(os.path.join(self.val_dir, "*.npz"))
        
        print(f"\n[File Count]")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        
        # ランダムにサンプルを選んで品質チェック
        print(f"\n[Quality Check]")
        check_count = min(100, len(train_files))
        print(f"  Checking {check_count} random samples...")
        
        np.random.shuffle(train_files)
        
        issues = []
        occlusion_rates = []
        
        for npz_path in train_files[:check_count]:
            partial, complete = self.load_sample(npz_path)
            
            # 各種チェック
            # 1. NaN/Inf チェック
            if np.any(np.isnan(partial)) or np.any(np.isnan(complete)):
                issues.append(f"{npz_path}: Contains NaN")
            if np.any(np.isinf(partial)) or np.any(np.isinf(complete)):
                issues.append(f"{npz_path}: Contains Inf")
            
            # 2. shape チェック
            if partial.shape != complete.shape:
                issues.append(f"{npz_path}: Shape mismatch")
            
            # 3. partial は complete の部分集合であることを確認
            partial_mask = partial > 0
            if not np.allclose(partial[partial_mask], complete[partial_mask], rtol=1e-5):
                issues.append(f"{npz_path}: Partial is not subset of complete")
            
            # 4. 欠損率を記録
            occlusion_rate = np.sum(partial == 0) / len(partial)
            occlusion_rates.append(occlusion_rate)
        
        # 結果を表示
        if len(issues) == 0:
            print(f"  ✓ All {check_count} samples passed quality check")
        else:
            print(f"  ✗ Found {len(issues)} issues:")
            for issue in issues[:10]:  # 最初の10個のみ表示
                print(f"    - {issue}")
        
        # 欠損率の統計
        if len(occlusion_rates) > 0:
            print(f"\n[Occlusion Rate Statistics]")
            print(f"  Mean: {np.mean(occlusion_rates):.3f}")
            print(f"  Std: {np.std(occlusion_rates):.3f}")
            print(f"  Min: {np.min(occlusion_rates):.3f}")
            print(f"  Max: {np.max(occlusion_rates):.3f}")
    
    def visualize_samples(
        self, 
        num_samples: int = 5,
        output_dir: Optional[str] = None
    ):
        """
        サンプルを可視化
        
        Args:
            num_samples: 可視化するサンプル数
            output_dir: 画像の保存先（Noneの場合は表示のみ）
        """
        print("\n" + "=" * 60)
        print("Sample Visualization")
        print("=" * 60)
        
        # train から num_samples 個をランダムに選択
        train_files = glob.glob(os.path.join(self.train_dir, "*.npz"))
        np.random.shuffle(train_files)
        sample_files = train_files[:num_samples]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for idx, npz_path in enumerate(sample_files):
            partial, complete = self.load_sample(npz_path)
            
            # 可視化
            self._plot_sample(
                partial, 
                complete, 
                title=f"Sample {idx+1}: {os.path.basename(npz_path)}",
                save_path=os.path.join(output_dir, f"sample_{idx+1}.png") if output_dir else None
            )
        
        if not output_dir:
            plt.show()
    
    def _plot_sample(
        self,
        partial: np.ndarray,
        complete: np.ndarray,
        title: str = "",
        save_path: Optional[str] = None
    ):
        """
        1つのサンプルをプロット
        
        Args:
            partial: 欠損あり
            complete: 完全
            title: グラフのタイトル
            save_path: 保存先（Noneの場合は保存しない）
        """
        n_beams = len(partial)
        angles = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. Polar plot - Complete
        ax = axes[0, 0]
        ax = plt.subplot(2, 2, 1, projection='polar')
        ax.plot(angles, complete, 'b-', linewidth=1, label='Complete')
        ax.set_title('Complete Scan', fontsize=12)
        ax.set_ylim(0, np.max(complete) * 1.1)
        ax.legend()
        ax.grid(True)
        
        # 2. Polar plot - Partial
        ax = plt.subplot(2, 2, 2, projection='polar')
        # partial > 0 の点のみプロット
        partial_angles = angles[partial > 0]
        partial_distances = partial[partial > 0]
        ax.scatter(partial_angles, partial_distances, c='r', s=5, alpha=0.6, label='Partial')
        ax.plot(angles, complete, 'b-', linewidth=0.5, alpha=0.3, label='Complete (ref)')
        ax.set_title('Partial Scan (with occlusion)', fontsize=12)
        ax.set_ylim(0, np.max(complete) * 1.1)
        ax.legend()
        ax.grid(True)
        
        # 3. Cartesian plot - Distance vs Angle
        ax = axes[1, 0]
        ax.plot(np.degrees(angles), complete, 'b-', linewidth=1, label='Complete', alpha=0.7)
        ax.scatter(
            np.degrees(angles)[partial > 0], 
            partial[partial > 0], 
            c='r', 
            s=10, 
            label='Partial',
            zorder=5
        )
        ax.set_xlabel('Angle [degrees]', fontsize=10)
        ax.set_ylabel('Distance [m]', fontsize=10)
        ax.set_title('Distance vs Angle', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Cartesian plot - XY view
        ax = axes[1, 1]
        # Complete を XY 座標に変換
        complete_x = complete * np.cos(angles)
        complete_y = complete * np.sin(angles)
        ax.scatter(complete_x, complete_y, c='b', s=5, alpha=0.3, label='Complete')
        
        # Partial を XY 座標に変換
        partial_mask = partial > 0
        partial_x = partial[partial_mask] * np.cos(angles[partial_mask])
        partial_y = partial[partial_mask] * np.sin(angles[partial_mask])
        ax.scatter(partial_x, partial_y, c='r', s=10, alpha=0.8, label='Partial')
        
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.set_title('Bird\'s Eye View', fontsize=12)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 統計情報を追加
        occlusion_rate = np.sum(partial == 0) / len(partial)
        valid_points_complete = np.sum(complete > 0)
        valid_points_partial = np.sum(partial > 0)
        
        stats_text = (
            f"Stats:\n"
            f"  Beams: {n_beams}\n"
            f"  Complete points: {valid_points_complete}\n"
            f"  Partial points: {valid_points_partial}\n"
            f"  Occlusion rate: {occlusion_rate:.2%}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {save_path}")
            plt.close()
    
    def plot_occlusion_distribution(self, output_path: Optional[str] = None):
        """
        欠損率の分布をプロット
        
        Args:
            output_path: 保存先（Noneの場合は表示のみ）
        """
        print("\n" + "=" * 60)
        print("Occlusion Rate Distribution")
        print("=" * 60)
        
        # train と val の欠損率を収集
        train_rates = []
        val_rates = []
        
        for npz_path in glob.glob(os.path.join(self.train_dir, "*.npz")):
            partial, complete = self.load_sample(npz_path)
            occlusion_rate = np.sum(partial == 0) / len(partial)
            train_rates.append(occlusion_rate)
        
        for npz_path in glob.glob(os.path.join(self.val_dir, "*.npz")):
            partial, complete = self.load_sample(npz_path)
            occlusion_rate = np.sum(partial == 0) / len(partial)
            val_rates.append(occlusion_rate)
        
        # プロット
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train
        ax = axes[0]
        ax.hist(train_rates, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Occlusion Rate', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Train Set (n={len(train_rates)})', fontsize=14)
        ax.axvline(np.mean(train_rates), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(train_rates):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Val
        ax = axes[1]
        ax.hist(val_rates, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Occlusion Rate', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Val Set (n={len(val_rates)})', fontsize=14)
        ax.axvline(np.mean(val_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(val_rates):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_path}")
            plt.close()
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and validate 2D LiDAR Point Completion Dataset"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for visualization images (default: None = show only)"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate dataset without visualization"
    )
    
    args = parser.parse_args()
    
    visualizer = DatasetVisualizer(args.dataset_root)
    
    # データセットを検証
    visualizer.validate_dataset()
    
    if not args.validate_only:
        # サンプルを可視化
        visualizer.visualize_samples(
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )
        
        # 欠損率の分布をプロット
        if args.output_dir:
            occlusion_plot_path = os.path.join(args.output_dir, "occlusion_distribution.png")
        else:
            occlusion_plot_path = None
        visualizer.plot_occlusion_distribution(output_path=occlusion_plot_path)


if __name__ == "__main__":
    main()