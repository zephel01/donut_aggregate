#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
動作確認用テストスクリプト
GPU検出、OCRエンジン、並列処理の動作を確認する
"""

import sys
from pathlib import Path


def test_gpu_detection():
    """GPU検出のテスト"""
    print("=" * 50)
    print("1. GPU検出テスト")
    print("=" * 50)

    try:
        from gpu_detector import GPUDetector

        gpu_type, gpu_info = GPUDetector.detect()
        print(f"✓ GPU検出成功!")
        print(f"  タイプ: {gpu_type.value}")
        print(f"  詳細: {gpu_info}")
        return True

    except Exception as e:
        print(f"✗ GPU検出失敗: {e}")
        return False


def test_ocr_engine_creation():
    """OCRエンジン作成のテスト"""
    print("\n" + "=" * 50)
    print("2. OCRエンジン作成テスト")
    print("=" * 50)

    try:
        from gpu_detector import GPUDetector, GPUType
        from ocr_engine import create_ocr_engine

        gpu_type, gpu_info = GPUDetector.detect()
        ocr_engine = create_ocr_engine(gpu_type)

        print(f"✓ OCRエンジン作成成功!")
        print(f"  エンジン: {ocr_engine.__class__.__name__}")
        print(f"  GPU使用: {ocr_engine.use_gpu}")
        return True

    except Exception as e:
        print(f"✗ OCRエンジン作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processing():
    """並列処理のテスト"""
    print("\n" + "=" * 50)
    print("3. 並列処理テスト")
    print("=" * 50)

    try:
        from gpu_detector import GPUDetector
        from ocr_engine import create_ocr_engine
        from parallel_ocr import ParallelOCREngine

        gpu_type, gpu_info = GPUDetector.detect()
        ocr_engine = create_ocr_engine(gpu_type)

        parallel_engine = ParallelOCREngine(
            ocr_engine=ocr_engine,
            num_workers=2,
            use_threading=ocr_engine.use_gpu
        )

        print(f"✓ 並列処理エンジン作成成功!")
        print(f"  ワーカー数: {parallel_engine.num_workers}")
        print(f"  スレッドプール: {parallel_engine.use_threading}")
        return True

    except Exception as e:
        print(f"✗ 並列処理エンジン作成失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_on_sample_image():
    """サンプル画像でのOCRテスト"""
    print("\n" + "=" * 50)
    print("4. サンプル画像OCRテスト")
    print("=" * 50)

    # サンプル画像を検索
    sample_images = []
    data_dir = Path("./data")

    if data_dir.exists():
        for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            sample_images.extend(data_dir.rglob(f"*{ext}"))

    if not sample_images:
        print("⊘ サンプル画像が見つかりません (./data に画像を配置してください)")
        return None

    # 最初の画像のみテスト
    sample_image = sample_images[0]
    print(f"  テスト画像: {sample_image}")

    try:
        from gpu_detector import GPUDetector
        from ocr_engine import create_ocr_engine

        gpu_type, gpu_info = GPUDetector.detect()
        ocr_engine = create_ocr_engine(gpu_type)

        # 画像読み込みとOCR
        import cv2
        img_bgr = cv2.imread(str(sample_image))

        if img_bgr is None:
            print(f"✗ 画像読み込み失敗: {sample_image}")
            return False

        # 前処理
        scale = 2.0
        img = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCR
        texts = ocr_engine.readtext(gray, detail=0, paragraph=True)

        if texts:
            print(f"✓ OCR成功!")
            print(f"  抽出されたテキスト:")
            for text in texts[:3]:  # 最初の3行のみ表示
                print(f"    - {text[:100]}...")
            if len(texts) > 3:
                print(f"    (他 {len(texts) - 3} 行)")
            return True
        else:
            print(f"✗ OCR成功ですがテキストが抽出されませんでした")
            return False

    except Exception as e:
        print(f"✗ OCR失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """全テストを実行"""
    print("\n" + "=" * 50)
    print("Donut OCR テストスイート")
    print("=" * 50)

    results = []

    # テスト実行
    results.append(("GPU検出", test_gpu_detection()))
    results.append(("OCRエンジン作成", test_ocr_engine_creation()))
    results.append(("並列処理", test_parallel_processing()))
    results.append(("サンプル画像OCR", test_ocr_on_sample_image()))

    # 結果集計
    print("\n" + "=" * 50)
    print("テスト結果サマリ")
    print("=" * 50)

    passed = 0
    failed = 0
    skipped = 0

    for test_name, result in results:
        if result is True:
            print(f"✓ {test_name}: PASS")
            passed += 1
        elif result is False:
            print(f"✗ {test_name}: FAIL")
            failed += 1
        else:
            print(f"⊘ {test_name}: SKIP")
            skipped += 1

    print(f"\n合計: {len(results)} テスト")
    print(f"  パス: {passed}")
    print(f"  失敗: {failed}")
    print(f"  スキップ: {skipped}")

    if failed == 0 and skipped == 0:
        print("\n🎉 全テストに合格しました!")
        return 0
    elif failed == 0:
        print("\n✓ 全テストが成功しました（一部スキップあり）")
        return 0
    else:
        print(f"\n⚠️  {failed}件のテストが失敗しました")
        return 1


if __name__ == "__main__":
    sys.exit(main())
