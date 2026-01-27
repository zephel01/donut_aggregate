# Donut OCR & Aggregate Tool

ゲーム画面のスクリーンショットからドーナツの情報をOCR抽出し、パワーや確率を集計するツールです。
特に日本語の認識と、アイテム名・パワー（種別・レベル）・エネルギーなどの詳細抽出に対応しています。

## 機能
- **OCR抽出**: 画像からテキストを読み取り、誤字補正を行います。
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

# パッケージのインストール
uv pip install -r requirements.txt
```

または、`uv sync` で一括構築も可能です。
```bash
uv sync
```

## 使い方

### OCR集計 (ocr_donut_aggregate.py)
`uv run` を使用してスクリプトを実行します。

```bash
uv run python ocr_donut_aggregate.py --input <画像ディレクトリ> --output-dir <出力ディレクトリ>
```

#### 実行例
```bash
uv run python ocr_donut_aggregate.py --input ./data --output-dir ./result
```

- `--input`: 画像ファイルが入っているディレクトリ（必須）
- `--output-dir`: 結果CSVを保存するディレクトリ（デフォルト: カレントディレクトリ）

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
