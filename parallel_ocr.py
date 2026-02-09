#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
並列OCR処理モジュール
multiprocessing を使用してOCR処理を並列化する
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import traceback

import cv2
import numpy as np

from ocr_engine import BaseOCREngine


class ParallelOCREngine:
    """並列OCR処理エンジン"""

    def __init__(
        self,
        ocr_engine: BaseOCREngine,
        num_workers: Optional[int] = None,
        batch_size: int = 1,
        use_threading: bool = False
    ):
        """
        並列OCRエンジンを初期化

        Args:
            ocr_engine: 使用するOCRエンジン
            num_workers: ワーカー数（NoneでCPUコア数）
            batch_size: バッチサイズ
            use_threading: スレッドプールを使用するか（GPU使用時推奨）
        """
        self.ocr_engine = ocr_engine
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.batch_size = batch_size
        self.use_threading = use_threading

    def process_images(
        self,
        image_paths: List[Path],
        show_progress: bool = True
    ) -> List[Tuple[Path, str]]:
        """
        画像リストを並列処理でOCRする

        Args:
            image_paths: 画像パスのリスト
            show_progress: 進捗バーを表示するか

        Returns:
            List[Tuple[Path, str]]: (画像パス, OCRテキスト) のリスト
        """
        total_images = len(image_paths)

        if show_progress:
            from tqdm import tqdm
            progress = tqdm(total=total_images, desc="OCR処理中")
        else:
            progress = None

        results = []
        errors = []

        # バッチ処理
        batches = [
            image_paths[i:i + self.batch_size]
            for i in range(0, total_images, self.batch_size)
        ]

        if self.use_threading:
            # GPU使用時はスレッドプール推奨（GIL解放される）
            executor_class = ThreadPoolExecutor
        else:
            # CPU使用時はプロセスプール推奨
            executor_class = ProcessPoolExecutor

        with executor_class(max_workers=self.num_workers) as executor:
            # 各バッチを処理する関数
            process_batch_func = partial(
                self._process_batch,
                preprocess_func=self._preprocess_image
            )

            # バッチを並列処理
            future_to_batch = {
                executor.submit(process_batch_func, batch): batch
                for batch in batches
            }

            # 結果を収集
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    error_msg = f"バッチ処理エラー: {e}\n{traceback.format_exc()}"
                    errors.append(error_msg)
                    # エラーが発生したバッチの画像を記録
                    for img_path in batch:
                        results.append((img_path, f"ERROR: {error_msg}"))

                if progress:
                    progress.update(len(batch))

        if progress:
            progress.close()

        if errors and show_progress:
            print(f"\n⚠️  {len(errors)}件のエラーが発生しました:")
            for err in errors[:5]:  # 最初の5件のみ表示
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... 他 {len(errors) - 5} 件")

        return results

    def _process_batch(
        self,
        batch: List[Path],
        preprocess_func
    ) -> List[Tuple[Path, str]]:
        """
        バッチ内の画像を順次処理

        Args:
            batch: 画像パスのリスト
            preprocess_func: 画像前処理関数

        Returns:
            List[Tuple[Path, str]]: (画像パス, OCRテキスト) のリスト
        """
        batch_results = []

        for img_path in batch:
            try:
                # 画像を読み込み
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    batch_results.append((img_path, "ERROR: 画像読み込み失敗"))
                    continue

                # 前処理
                img = preprocess_func(img_bgr)

                # OCR
                texts = self.ocr_engine.readtext(img, detail=0, paragraph=True)
                text = "\n".join(texts)

                batch_results.append((img_path, text))

            except Exception as e:
                error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
                batch_results.append((img_path, error_msg))

        return batch_results

    @staticmethod
    def _preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
        """
        画像の前処理

        Args:
            img_bgr: BGR画像

        Returns:
            np.ndarray: 前処理されたグレースケール画像
        """
        img = img_bgr.copy()

        # 1) 拡大（小さい文字対策）
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # 2) グレースケール
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return gray


def calculate_optimal_workers(
    use_gpu: bool,
    total_images: int,
    min_workers: int = 1,
    max_workers: Optional[int] = None
) -> int:
    """
    最適なワーカー数を計算

    Args:
        use_gpu: GPUを使用するか
        total_images: 画像の総数
        min_workers: 最小ワーカー数
        max_workers: 最大ワーカー数（NoneでCPUコア数）

    Returns:
        int: 最適なワーカー数
    """
    cpu_count = multiprocessing.cpu_count()

    if max_workers is None:
        max_workers = cpu_count

    if use_gpu:
        # GPU使用時はワーカー数を制限（GPUメモリの競合防止）
        return min(min_workers + 1, max_workers)
    else:
        # CPU使用時はコア数に応じて調整
        # Apple Siliconは高コア数CPUなので、コア数の75%程度を使用
        optimal = int(cpu_count * 0.75)

        # max_workers以下に収める
        optimal = min(optimal, max_workers)

        # 画像数が少ない場合はワーカー数を減らす
        if total_images < optimal:
            optimal = total_images

        return max(optimal, min_workers)


def benchmark_processing(
    ocr_engine: BaseOCREngine,
    image_paths: List[Path],
    num_workers_list: List[int] = None
) -> Dict[int, float]:
    """
    異なるワーカー数での処理時間をベンチマーク

    Args:
        ocr_engine: 使用するOCRエンジン
        image_paths: 画像パスのリスト
        num_workers_list: ワーカー数のリスト（Noneで自動生成）

    Returns:
        Dict[int, float]: ワーカー数 -> 処理時間（秒）
    """
    if num_workers_list is None:
        num_workers_list = [1, 2, 4, 8]
        cpu_count = multiprocessing.cpu_count()
        if cpu_count > 8:
            num_workers_list.append(cpu_count)

    results = {}

    # 最初の10枚のみでベンチマーク
    test_images = image_paths[:10]

    print("ベンチマーク開始...")
    for num_workers in num_workers_list:
        engine = ParallelOCREngine(
            ocr_engine=ocr_engine,
            num_workers=num_workers,
            use_threading=ocr_engine.use_gpu
        )

        start_time = time.time()
        engine.process_images(test_images, show_progress=False)
        elapsed = time.time() - start_time

        results[num_workers] = elapsed
        print(f"  ワーカー数 {num_workers}: {elapsed:.2f}秒")

    # 最適なワーカー数を推奨
    best_workers = min(results, key=results.get)
    print(f"\n推奨ワーカー数: {best_workers} ({results[best_workers]:.2f}秒)")

    return results


def main():
    """テスト用: 並列処理の動作確認"""
    from gpu_detector import GPUDetector, GPUType
    from ocr_engine import create_ocr_engine

    # GPU検出
    gpu_type, gpu_info = GPUDetector.detect()
    print(f"検出されたGPU: {gpu_type.value}")
    print(f"詳細: {gpu_info}\n")

    # OCRエンジン作成
    ocr_engine = create_ocr_engine(gpu_type)
    print(f"使用するOCRエンジン: {ocr_engine.__class__.__name__}\n")

    # テスト画像を検索
    from pathlib import Path
    test_dir = Path("./data")
    if not test_dir.exists():
        print("テスト画像ディレクトリが見つかりません: ./data")
        return

    image_paths = []
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        image_paths.extend(test_dir.rglob(f"*{ext}"))

    if not image_paths:
        print("画像が見つかりません")
        return

    print(f"見つかった画像: {len(image_paths)}枚\n")

    # ベンチマーク実行（最初の10枚）
    print("ベンチマーク実行中...")
    benchmark_processing(ocr_engine, image_paths)

    # 最適なワーカー数で全画像を処理
    print("\n全画像のOCR処理を開始します...")
    optimal_workers = calculate_optimal_workers(
        use_gpu=ocr_engine.use_gpu,
        total_images=len(image_paths)
    )

    parallel_engine = ParallelOCREngine(
        ocr_engine=ocr_engine,
        num_workers=optimal_workers,
        use_threading=ocr_engine.use_gpu
    )

    start_time = time.time()
    results = parallel_engine.process_images(image_paths)
    elapsed = time.time() - start_time

    print(f"\n完了! {len(results)}枚の画像を {elapsed:.2f}秒で処理しました")
    print(f"平均: {elapsed / len(results):.3f}秒/枚")


if __name__ == "__main__":
    main()
