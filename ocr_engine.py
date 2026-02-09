#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCRエンジン抽象化レイヤー
EasyOCR を使用
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Any
import numpy as np

from gpu_detector import GPUType


class BaseOCREngine(ABC):
    """OCRエンジンの基底クラス"""

    def __init__(self, languages: List[str], use_gpu: bool = False):
        self.languages = languages
        self.use_gpu = use_gpu
        self._reader = None

    @abstractmethod
    def initialize(self) -> None:
        """OCRエンジンを初期化する"""
        pass

    @abstractmethod
    def readtext(
        self,
        image: np.ndarray,
        detail: int = 0,
        paragraph: bool = False,
        **kwargs
    ) -> List[str]:
        """
        画像からテキストを読み取る

        Args:
            image: 画像配列 (numpy array)
            detail: 詳細出力レベル (0: テキストのみ, 1: バウンディングボックス付き)
            paragraph: 段落として結合するかどうか
            **kwargs: エンジン固有のパラメータ

        Returns:
            List[str]: 抽出されたテキストのリスト
        """
        pass

    @property
    def reader(self) -> Any:
        """内部のリーダーオブジェクト"""
        if self._reader is None:
            self.initialize()
        return self._reader


class EasyOCREngine(BaseOCREngine):
    """EasyOCR エンジン"""

    def initialize(self) -> None:
        """EasyOCR リーダーを初期化"""
        try:
            import easyocr
            import torch
        except ImportError as e:
            raise ImportError(
                f"必要なライブラリがインストールされていません: {e}"
            )

        # デバイスの決定
        device = 'cpu'
        use_mps = False

        if self.use_gpu:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
                use_mps = True
            else:
                print("警告: CUDA/MPS が利用可能ではありません。CPUを使用します。")

        # EasyOCRリーダーを初期化
        # MPSの場合は、まずCPUで初期化してからモデルをMPSに移動
        if use_mps:
            print("MPS (Apple Silicon) モードでEasyOCRを初期化します...")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,  # 一時的にCPUで初期化
                verbose=False
            )

            # モデルをMPSデバイスに移動
            print("  モデルをMPSデバイスに移動中...")
            try:
                # DetectorをMPSに移動
                if hasattr(self._reader.detector, 'model'):
                    if hasattr(self._reader.detector, 'module'):
                        # DataParallelを使用している場合
                        self._reader.detector.module.model = self._reader.detector.module.model.to('mps')
                    else:
                        self._reader.detector.model = self._reader.detector.model.to('mps')

                # RecognizerをMPSに移動
                if hasattr(self._reader.recognizer, 'model'):
                    if hasattr(self._reader.recognizer, 'module'):
                        # DataParallelを使用している場合
                        self._reader.recognizer.module.model = self._reader.recognizer.module.model.to('mps')
                    else:
                        self._reader.recognizer.model = self._reader.recognizer.model.to('mps')

                self.device = 'mps'
                print("  ✓ MPSデバイスへの移動完了")
            except Exception as e:
                print(f"  ✗ MPSデバイスへの移動に失敗: {e}")
                print("  CPUフォールバックを使用します")
                self.device = 'cpu'
        elif device == 'cuda':
            # CUDAを使用
            self._reader = easyocr.Reader(
                self.languages,
                gpu=True,
                verbose=False
            )
            self.device = 'cuda'
        else:
            # CPUを使用
            self._reader = easyocr.Reader(
                self.languages,
                gpu=False,
                verbose=False
            )
            self.device = 'cpu'

    def readtext(
        self,
        image: np.ndarray,
        detail: int = 0,
        paragraph: bool = False,
        **kwargs
    ) -> List[str]:
        """EasyOCR でテキストを読み取る"""
        return self.reader.readtext(image, detail=detail, paragraph=paragraph, **kwargs)


class RapidOCREngine(BaseOCREngine):
    """RapidOCR エンジン (Apple Silicon向け高速版)"""

    def initialize(self) -> None:
        """RapidOCR リーダーを初期化"""
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "RapidOCR がインストールされていません。"
                "インストールしてください: uv pip install rapidocr_onnxruntime"
            )

        self._reader = RapidOCR()

    def readtext(
        self,
        image: np.ndarray,
        detail: int = 0,
        paragraph: bool = False,
        **kwargs
    ) -> List[str]:
        """RapidOCR でテキストを読み取る"""
        # RapidOCR は [[[box], text, score], ...] の形式で返す
        # スコア閾値を下げてより多くのテキストを抽出
        result, _ = self.reader(image, cls=True, det_db_thresh=0.3, drop_score=0.3)

        if not result:
            return []

        if detail == 0:
            # テキストのみ (result[1] がテキスト)
            texts = [item[1] for item in result]
            return texts
        else:
            # バウンディングボックス付き
            return result


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR エンジン (AMD ROCm向け)"""

    def initialize(self) -> None:
        """PaddleOCR リーダーを初期化"""
        try:
            from paddleocr import PaddleOCR
            import paddle
        except ImportError as e:
            raise ImportError(
                f"PaddleOCRまたはPaddlePaddleがインストールされていません: {e}"
            )

        # GPUを使用する場合、PaddlePaddleのデバイスを設定
        if self.use_gpu:
            try:
                # PaddlePaddleの情報を確認
                import paddle
                print(f"  PaddlePaddle version: {paddle.__version__}")
                print(f"  CUDA available: {paddle.is_compiled_with_cuda()}")
                print(f"  ROCm available: {hasattr(paddle, 'is_compiled_with_rocm')}")
                if hasattr(paddle, 'is_compiled_with_rocm'):
                    print(f"  is_compiled_with_rocm(): {paddle.is_compiled_with_rocm()}")
                print(f"  GPU available: {paddle.is_compiled_with_cuda() or (hasattr(paddle, 'is_compiled_with_rocm') and paddle.is_compiled_with_rocm())}")
                print(f"  GPU device count: {paddle.device_count() if hasattr(paddle, 'device_count') else 'N/A'}")

                # ROCmまたはCUDAを確認
                if hasattr(paddle, 'is_compiled_with_rocm') and paddle.is_compiled_with_rocm():
                    # ROCm環境
                    paddle.set_device('gpu:0')
                    self.device = 'gpu'
                    print("  ROCm (AMD GPU) を使用します")
                elif paddle.is_compiled_with_cuda():
                    # CUDA環境
                    paddle.set_device('gpu:0')
                    self.device = 'gpu'
                    print("  CUDA (NVIDIA GPU) を使用します")
                else:
                    # GPUが利用不可
                    print("  警告: PaddlePaddleはGPUでコンパイルされていません。CPUを使用します。")
                    self.device = 'cpu'
            except Exception as e:
                print(f"  警告: GPU設定に失敗しました。CPUを使用します: {e}")
                import traceback
                traceback.print_exc()
                self.device = 'cpu'
        else:
            self.device = 'cpu'

        # use_angle_cls=True で文字角度補正を有効化
        self._reader = PaddleOCR(
            use_angle_cls=True,
            lang='japan',  # 日本語
        )

    def readtext(
        self,
        image: np.ndarray,
        detail: int = 0,
        paragraph: bool = False,
        **kwargs
    ) -> List[str]:
        """EasyOCR でテキストを読み取る"""
        return self.reader.readtext(image, detail=detail, paragraph=paragraph, **kwargs)

def create_ocr_engine(
    gpu_type: GPUType,
    languages: Optional[List[str]] = None,
    force_engine: Optional[str] = None,
    use_gpu: Optional[bool] = None
) -> BaseOCREngine:
    """
    EasyOCRエンジンを作成する

    Args:
        gpu_type: 検出されたGPUタイプ（ROCmもEasyOCRを使用）
        languages: 使用する言語リスト (デフォルト: ['ja', 'en'])
        force_engine: エンジンの強制指定（現在はeasyocrのみ対応）
        use_gpu: GPUを使用するかどうか (Noneで自動判定)

    Returns:
        BaseOCREngine: OCRエンジンのインスタンス
    """
    if languages is None:
        languages = ['ja', 'en']

    # GPU使用の自動判定
    if use_gpu is None:
        use_gpu = gpu_type != GPUType.CPU

    # force_engineのチェック（easyocrのみ対応）
    if force_engine and force_engine != 'easyocr':
        raise ValueError(
            f"不明なエンジン: {force_engine}. "
            f"選択可能: ['easyocr']"
        )

    # EasyOCRを使用（全GPUタイプ対応）
    return EasyOCREngine(languages=languages, use_gpu=use_gpu)


def main():
    """テスト用: OCRエンジンの動作確認"""
    from gpu_detector import GPUDetector

    gpu_type, gpu_info = GPUDetector.detect()
    print(f"検出されたGPU: {gpu_type.value}")
    print(f"詳細: {gpu_info}")
    print()

    # エンジンを作成
    engine = create_ocr_engine(gpu_type)
    print(f"使用するOCRエンジン: {engine.__class__.__name__}")

    # テスト画像がある場合のみテスト
    from pathlib import Path
    test_image = Path("./test_image.jpg")
    if test_image.exists():
        print(f"\nテスト画像: {test_image}")
        import cv2
        img = cv2.imread(str(test_image))
        texts = engine.readtext(img, detail=0, paragraph=True)
        print("抽出されたテキスト:")
        for text in texts:
            print(f"  - {text}")
    else:
        print("\nテスト画像が見つかりません: ./test_image.jpg")


if __name__ == "__main__":
    main()
