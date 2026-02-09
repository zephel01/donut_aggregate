# Donut OCR & Aggregate Tool

ゲーム画面のスクリーンショットからドーナツの情報をOCR抽出し、パワーや確率を集計するツールです。
特に日本語の認識と、アイテム名・パワー（種別・レベル）・エネルギーなどの詳細抽出に対応しています。

## 機能
- **OCR抽出**: 画像からテキストを読み取り、誤字補正を行います。
- **GPU対応**: NVIDIA CUDA、Apple Silicon (MPS)、AMD ROCmに対応
- **並列処理**: multiprocessingでCPU/GPUを効率的に活用
- **データ抽出**:
  - ドーナツ名
  - パワー（例: `どうぐパワー: ボール Lv.3`）
  - エネルギー (kcal)
  - その他属性 (+Lv, スターランクなど)
- **カテゴリ集計**:
  - パワーを5つのフレーバー（Sweet, Spicy, Sour, Bitter, Fresh）に分類して集計
  - パワーの出現確率を計算
- **組み合わせ集計**:
  - **どうぐ + どっさり**: `どうぐパワー` と `どっさりパワー` の組み合わせを抽出
  - **スイート組み合わせ**: `Sweet` カテゴリ内の全パワー（かがやき、オヤブン等）を結合して抽出
- **自動保存**: 日時付きのファイル名で保存

## パフォーマンス

2000枚の画像の処理時間目安：

| 環境 | モード | 予想時間 |
|------|--------|----------|
| CPU直列 | EasyOCR | 約66分 |
| CPU並列(4コア) | EasyOCR | 約17分 |
| **NVIDIA GPU** | EasyOCR + 並列 | **約3〜5分** |
| **Apple Silicon** | RapidOCR + 並列 | **約3〜5分** |
| **AMD ROCm** | PaddleOCR + 並列 | **約5〜7分** |

## セットアップ方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/[YOUR_USERNAME]/donut_aggregate.git
cd donut_aggregate
```

### 2. 環境構築
高速なパッケージマネージャー `uv` を使用します。

- `uv` の導入: [公式ドキュメント](https://github.com/astral-sh/uv)

```bash
# 仮想環境の作成
uv venv

# 仮想環境の有効化
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

#### GPU別のセットアップ

**CPUのみ（標準）:**
```bash
uv pip install -r requirements.txt
```

**Apple Silicon (MPS):**
```bash
uv pip install -r requirements.txt -r requirements-apple-silicon.txt
```

**NVIDIA GPU (CUDA):**
```bash
# PyTorch (CUDA) を先にインストール
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

**AMD GPU (ROCm, Linuxのみ):**
```bash
# 事前にROCm環境のセットアップが必要（詳細は下記「ROCm環境セットアップ」を参照）
uv pip install -r requirements.txt -r requirements-amd-rocm.txt
```

---

## ROCm環境セットアップ（AMD GPU）

### 対応GPUとOS

**対応GPU:**
- AMD AI MAX 395+ (MI325X, MI300Xなど)
- Radeon RX 6000/7000シリーズ
- Radeon Proシリーズ
- Instinctシリーズ

対応状況: [AMD ROCmサポートGPU](https://rocm.docs.amd.com/en/latest/deploy/linux/requirements.html)

**対応OS:**
- Ubuntu 20.04 / 22.04（推奨）
- CentOS / RHEL 7.9 / 8.5
- Fedora 34+
- SLES 15 SP4

⚠️ **macOS / Windows はサポートされていません**

---

### 手順1: ROCmのインストール

#### Ubuntuの場合（推奨）

```bash
# 1. システムを更新
sudo apt update
sudo apt upgrade -y

# 2. AMDGPUドライバのインストール
sudo apt install amdgpu-install

# 3. ROCmのインストール（最新版）
sudo amdgpu-install --usecase=rocm,hip --no-dkms

# 4. ユーザーをrenderグループとvideoグループに追加
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# 5. 再ログインしてグループ設定を反映
# ログアウトして再ログインしてください
```

#### CentOS/RHELの場合

```bash
# 1. リポジトリの追加
sudo yum install -y https://repo.radeon.com/amdgpu-install/6.0/amdgpu-install-6.0.60000-1.el7.noarch.rpm

# 2. ROCmのインストール
sudo amdgpu-install --usecase=rocm,hip --no-dkms

# 3. ユーザーをrenderグループとvideoグループに追加
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# 4. 再ログインしてグループ設定を反映
```

---

### 手順2: 環境変数の設定

`~/.bashrc` または `~/.zshrc` に以下を追加：

```bash
# ROCm環境変数
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HIP_PATH=/opt/rocm
export ROCM_PATH=/opt/rocm
```

設定を反映：

```bash
source ~/.bashrc
```

---

### 手順3: ROCmの動作確認

```bash
# ROCmのバージョン確認
/opt/rocm/bin/rocminfo

# GPUの状態確認
/opt/rocm/bin/rocm-smi
```

出力例：

```
ROCm Version: 6.0.0
...
GPU0    MI325X        0x7449
...
```

---

### 手順4: PyTorch (ROCm版) のインストール

```bash
# ROCm 5.7用のPyTorch（推奨）
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# uvを使用する場合
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# または、ROCm 6.0用のPyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

#### PyTorch (ROCm版) の動作確認

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'HIP available: {torch.version.hip is not None}')
print(f'HIP version: {torch.version.hip}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    # テンソル演算テスト
    x = torch.randn(3, 3).cuda()
    print(f'GPU tensor test: {x.sum()}')
"
```

出力例：

```
PyTorch version: 2.2.0+rocm5.7
CUDA available: True
HIP available: True
HIP version: 5.7.31104
GPU count: 1
GPU name: AMD Instinct MI325X
GPU tensor test: tensor(0.1234, device='cuda:0')
```

---

### 手順5: PaddlePaddle (ROCm版) のインストール

```bash
# ROCm版PaddlePaddleのインストール
pip install paddlepaddle-gpu

# uvを使用する場合
uv pip install paddlepaddle-gpu
```

---

### 手順6: OCRパッケージのインストール

```bash
# プロジェクトのディレクトリに移動
cd /path/to/donut_aggregate

# 基本パッケージのインストール
uv pip install -r requirements.txt

# ROCm用の追加パッケージ
uv pip install -r requirements-amd-rocm.txt
```

---

### 手順7: テスト実行

```bash
# テストスクリプトで動作確認
uv run python test_ocr.py
```

出力例：

```
==================================================
Donut OCR テストスイート
==================================================
==================================================
1. GPU検出テスト
==================================================
✓ GPU検出成功!
  タイプ: amd_rocm
  詳細: AMD GPU (ROCm via PyTorch)
...
```

---

## ROCm環境のトラブルシューティング

### ROCmが認識されない

```bash
# ドライバのログ確認
dmesg | grep -i amdgpu

# ROCmの再インストール
sudo amdgpu-install --uninstall
sudo amdgpu-install --usecase=rocm,hip --no-dkms

# 再起動
sudo reboot
```

### PyTorchがROCmを認識しない

```bash
# 環境変数の確認
echo $HIP_PATH
echo $LD_LIBRARY_PATH

# 手動で環境変数を設定して実行
HIP_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib python -c "import torch; print(torch.cuda.is_available())"
```

### GPUメモリ不足

```bash
# ワーカー数を減らして実行
uv run python ocr_donut_aggregate.py --input ./data --workers 2
```

### パーミッションエラー

```bash
# デバイスのパーミッションを確認
ls -la /dev/kfd
ls -la /dev/dri

# パーミッションの修正（必要な場合）
sudo chmod 666 /dev/kfd
sudo chmod 666 /dev/dri/*
```

---

### ROCmのアンインストール

```bash
# ROCmのアンインストール
sudo amdgpu-install --uninstall

# パッケージの完全削除
sudo apt remove --purge rocm* rocm-dev* miopen* hip* llvm-amdgpu* comgr* rocfft* rocBLAS* hipBLAS* rocprim* rocThrust* rocALUTION*

# 再起動
sudo reboot
```

---

## 使い方

### OCR集計 (ocr_donut_aggregate.py)

#### 基本実行
```bash
uv run python ocr_donut_aggregate.py --input <画像ディレクトリ> --output-dir <出力ディレクトリ>
```

#### GPU・並列処理オプション

```bash
# GPU自動検出（推奨）
uv run python ocr_donut_aggregate.py --input ./data --gpu auto

# NVIDIA GPUを指定
uv run python ocr_donut_aggregate.py --input ./data --gpu cuda

# Apple Siliconを指定
uv run python ocr_donut_aggregate.py --input ./data --gpu mps

# AMD ROCmを指定（Linuxのみ）
uv run python ocr_donut_aggregate.py --input ./data --gpu rocm

# CPUのみを使用
uv run python ocr_donut_aggregate.py --input ./data --gpu cpu

# OCRエンジンを強制指定
uv run python ocr_donut_aggregate.py --input ./data --engine rapidocr

# ワーカー数を指定（デフォルト: 自動）
uv run python ocr_donut_aggregate.py --input ./data --workers 4

# ベンチマークモード（最初の10枚で最適設定を探す）
uv run python ocr_donut_aggregate.py --input ./data --benchmark
```

#### オプション詳細

| オプション | 値 | 説明 |
|-----------|-----|------|
| `--gpu` | auto\|cuda\|mps\|rocm\|cpu | GPU使用モード（デフォルト: auto） |
| `--engine` | easyocr\|rapidocr\|paddleocr | OCRエンジン強制指定 |
| `--workers` | 数字 | ワーカー数（デフォルト: 自動） |
| `--benchmark` | - | ベンチマークモード |
| `--input` | パス | 画像ディレクトリ（必須） |
| `--output-dir` | パス | 出力ディレクトリ（デフォルト: カレント） |
| `--out` | ファイル名 | 抽出結果CSVファイル名 |
| `--summary` | ファイル名 | 集計サマリCSVファイル名 |

### タイムスタンプ修正 (after_timestamp_to_date.py)
ファイル名のタイムスタンプを元に、ファイルの作成・更新日時を修正します。
```bash
uv run python after_timestamp_to_date.py --input <画像ディレクトリ> --output <出力ディレクトリ>
```

## 出力ファイル

実行すると、指定したディレクトリ（例: `./result`）に以下の2つのファイルが生成されます。
ファイル名には実行時のタイムスタンプが含まれます。

1.  **抽出データ一覧** (`extracted_YYYYMMDD_HHMMSS.csv`)
    -   各画像のOCR結果と抽出されたフィールドが含まれます。
    -   `powers_summary` カラムには、抽出されたパワーがカテゴリごとに記載されます。
2.  **集計サマリ** (`summary_YYYYMMDD_HHMMSS.csv`)
    -   各パワーの出現数と出現確率（%）が集計されています。
    -   フレーバーカテゴリ別に見やすく整理されています。

## 動作確認

環境が正しくセットアップされたか確認するには、テストスクリプトを実行します。

```bash
uv run python test_ocr.py
```

このスクリプトは以下のテストを実行します：
1. GPU検出
2. OCRエンジン作成
3. 並列処理
4. サンプル画像でのOCR

## トラブルシューティング

### GPUが検出されない場合
```bash
# GPU情報を確認
python gpu_detector.py
```

### OCRエンジンのエラー
- **EasyOCR**: PyTorchが正しくインストールされているか確認
- **RapidOCR**: `uv pip install rapidocr-onnxruntime` を実行
- **PaddleOCR**: `uv pip install paddlepaddle paddleocr` を実行

### メモリ不足
- ワーカー数を減らす: `--workers 2`
- バッチサイズを調整（`parallel_ocr.py` の `batch_size`）

## プロジェクト構成

```
donut_aggregate/
├── ocr_donut_aggregate.py      # メインスクリプト
├── gpu_detector.py             # GPU検出モジュール
├── ocr_engine.py               # OCRエンジン抽象化
├── parallel_ocr.py              # 並列処理モジュール
├── test_ocr.py                 # テストスクリプト
├── requirements.txt            # 基本パッケージ
├── requirements-apple-silicon.txt  # Apple Silicon用
├── requirements-nvidia-cuda.txt     # NVIDIA GPU用
└── requirements-amd-rocm.txt        # AMD ROCm用
```