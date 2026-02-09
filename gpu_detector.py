#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU自動検出モジュール
NVIDIA (CUDA), Apple Silicon (MPS), AMD (ROCm), CPUを自動判定
"""

from __future__ import annotations

import platform
import subprocess
from enum import Enum
from typing import Optional, Tuple


class GPUType(Enum):
    """GPUタイプの列挙型"""
    NVIDIA_CUDA = "nvidia_cuda"
    APPLE_MPS = "apple_mps"
    AMD_ROCM = "amd_rocm"
    CPU = "cpu"


class GPUDetector:
    """GPU検出クラス"""

    @staticmethod
    def detect() -> Tuple[GPUType, str]:
        """
        利用可能なGPUを検出する

        Returns:
            Tuple[GPUType, str]: (GPUタイプ, 詳細情報)
        """
        # 1. NVIDIA GPU (CUDA) の検出
        nvidia_info = GPUDetector._detect_nvidia()
        if nvidia_info:
            return GPUType.NVIDIA_CUDA, nvidia_info

        # 2. Apple Silicon (MPS) の検出
        apple_info = GPUDetector._detect_apple_silicon()
        if apple_info:
            return GPUType.APPLE_MPS, apple_info

        # 3. AMD ROCm の検出（Linuxのみ）
        amd_info = GPUDetector._detect_amd_rocm()
        if amd_info:
            return GPUType.AMD_ROCM, amd_info

        # 4. CPU フォールバック
        return GPUType.CPU, GPUDetector._get_cpu_info()

    @staticmethod
    def _detect_nvidia() -> Optional[str]:
        """NVIDIA GPU (CUDA) を検出"""
        try:
            # nvidia-smi コマンドで検出
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                return f"NVIDIA {gpu_name}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # PyTorch がインストールされている場合は CUDA も確認
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return f"NVIDIA {gpu_name} (CUDA {torch.version.cuda})"
        except ImportError:
            pass

        return None

    @staticmethod
    def _detect_apple_silicon() -> Optional[str]:
        """Apple Silicon (MPS) を検出"""
        # macOS で ARM アーキテクチャか確認
        if platform.system() != "Darwin":
            return None

        if platform.machine() != "arm64":
            return None

        # PyTorch がインストールされている場合は MPS も確認
        try:
            import torch
            if torch.backends.mps.is_available():
                return "Apple Silicon (MPS)"
        except ImportError:
            pass

        # ARM アーキテクチャなら Apple Silicon とみなす
        # PyTorch 未インストールの場合も MPS が使用可能な可能性がある
        return "Apple Silicon (ARM64)"

    @staticmethod
    def _detect_amd_rocm() -> Optional[str]:
        """AMD ROCm を検出（Linuxのみ）"""
        if platform.system() != "Linux":
            return None

        # rocm-smi コマンドで検出
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return "AMD GPU (ROCm)"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # PyTorch がインストールされている場合は ROCm も確認
        try:
            import torch
            # ROCm の PyTorch でロードされているか確認
            if torch.version.hip is not None:
                return "AMD GPU (ROCm via PyTorch)"
        except ImportError:
            pass

        return None

    @staticmethod
    def _get_cpu_info() -> str:
        """CPU情報を取得"""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        return f"CPU ({cpu_count} cores)"


def main():
    """テスト用: 検出したGPU情報を表示"""
    gpu_type, gpu_info = GPUDetector.detect()
    print(f"検出されたGPU: {gpu_type.value}")
    print(f"詳細: {gpu_info}")
    print(f"\n推奨設定:")
    print("  OCRエンジン: EasyOCR")
    if gpu_type == GPUType.NVIDIA_CUDA:
        print("  デバイス: CUDA")
    elif gpu_type == GPUType.APPLE_MPS:
        print("  デバイス: MPS")
    elif gpu_type == GPUType.AMD_ROCM:
        print("  デバイス: ROCm")
    else:
        print("  デバイス: CPU")
        print("  並列処理: 有効")


if __name__ == "__main__":
    main()
