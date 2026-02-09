#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
画像から以下を抽出してCSV化＋集計する:
- スターランク (例: 5)
- どうぐパワー :ボール Lv.3
- どっさりパワー : Lv.2
- ほかくパワー : かくとう Lv.2
- プラスレベル +Lv.111
- はらもちエネルギー 4050kcal

使い方:
  python ocr_donut_aggregate.py --input ./images --output-dir ./result

GPUと並列処理オプション:
  --gpu auto|cuda|mps|rocm|cpu  GPU使用モード (デフォルト: auto)
  --workers N                     ワーカー数 (デフォルト: 自動)

要件:
  pip install easyocr opencv-python pandas tqdm
"""

from __future__ import annotations

import argparse
import re
import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any

import cv2
import pandas as pd
from tqdm import tqdm

# 新規モジュールのインポート
from gpu_detector import GPUDetector, GPUType
from ocr_engine import create_ocr_engine
from parallel_ocr import ParallelOCREngine, calculate_optimal_workers

# -----------------------------
# Constants
# -----------------------------
POWER_CATEGORIES = {
    "sweet": ["オヤブンパワー", "でかでかパワー", "ちびちびパワー", "かがやきパワー"],
    "spicy": ["こうげきパワー", "とくこうパワー", "すばやさパワー", "わざパワー"],
    "sour": [
        "どっさりパワー", "どうぐパワー", "メガパワー", "とくべつパワー",
        "きのみパワー", "アメパワー", "コインパワー", "おたからパワー"
    ],
    "bitter": ["めんえきパワー", "ぼうぎょパワー", "とくぼうパワー"],
    "fresh": ["ほかくパワー", "そうぐうパワー"]
}

# Flatten for reverse lookup
POWER_TO_CATEGORY = {}
for cat, names in POWER_CATEGORIES.items():
    for n in names:
        POWER_TO_CATEGORY[n] = cat


# -----------------------------
# OCR 前処理
# -----------------------------
def preprocess(img_bgr):
    """
    OCR精度を上げるための軽い前処理。
    画像によっては逆効果もあるので、必要なら調整してください。
    """
    img = img_bgr.copy()

    # 1) 拡大（小さい文字対策）
    scale = 2.0
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2) グレースケール
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray


# -----------------------------
# テキスト抽出（OCRエンジン使用）
# -----------------------------
def ocr_image(ocr_engine, image_path: Path) -> str:
    """OCRエンジンを使用して画像からテキストを抽出"""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"画像を読み込めません: {image_path}")

    proc = preprocess(img_bgr)

    # OCRエンジンを使用してテキスト抽出
    lines: List[str] = ocr_engine.readtext(proc, detail=0, paragraph=True)

    # 連結して1つの全文に
    text = "\n".join(lines)

    # OCRの揺れを少し正規化
    return normalize_ocr_text(text)


def normalize_ocr_text(text: str) -> str:
    """OCRテキストの正規化"""
    # 1) 記号の統一
    text = text.replace("：", ":").replace("＋", "+").replace("　", " ")

    # 2) 紛らわしいカタカナ・漢字の正規化
    text = re.sub(r"[バハパ][ワ7][ー一]", "パワー", text)
    text = re.sub(r"[とど][う][くぐ]", "どうぐ", text)
    text = text.replace("どつさり", "どっさり")
    text = text.replace("Lv.", "Lv").replace("Lv", "Lv.")

    # 3) 誤字補正
    text = correct_text(text)

    return text


def correct_text(text: str) -> str:
    """
    OCR特有の誤字脱字を辞書/ルールベースで修正する。
    """
    replacements = [
        # 英単語/ヘッダーまわり
        (r"IT[ew]m?", "Item"),
        (r"Elavor", "Flavor"),
        (r"kca[lt]?", "kcal"),
        (r"Domut", "Donut"),

        # カタカナ小文字化など
        (r"フレツミュ", "フレッシュ"),
        (r"ミツクス", "ミックス"),
        (r"スイト", "スイート"),
        (r"コンフイ", "コンフィ"),
        (r"エスバー", "エスパー"),
        (r"フエアリー", "フェアリー"),

        # パワー名/タイプまわりの誤字
        (r"[か力][か力がガか力][やきさ][きさ]パワー", "かがやきパワー"), # かガやき, かかやき, かかもき 等
        (r"ぜんふ", "ぜんぶ"),
        (r"すぺて", "すべて"),

        # ゲーム用語/文脈
        (r"連[過遍]", "遭遇"),
        (r"[渡燕]得", "捕獲"), # 仮置き

        # 数字/Lvまわり
        (r"Lv\.?1o9", "Lv.109"),
    ]

    for pat, rep in replacements:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    return text


# -----------------------------
# 欲しい項目のパース
# -----------------------------
def _find_int(patterns: List[str], text: str) -> Optional[int]:
    for p in patterns:
        m = re.search(p, text)
        if m:
            val_str = m.group(1)
            # o/O -> 0 などの補正
            val_str = val_str.replace("o", "0").replace("O", "0")
            try:
                return int(val_str)
            except ValueError:
                pass
    return None


def _find_str(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    return None


def parse_fields(text: str) -> Dict[str, Any]:
    """
    OCRテキストから情報を抽出する
    """
    # 1. Star Rank
    star = _find_int(
        [r"★\s*([0-9])", r"スター\s*ランク\s*([0-9])", r"スターランク\s*([0-9])", r"ランク\s*([0-9])"],
        text
    )

    # 2. Powers (Generic Extraction)
    extracted_powers = []

    # 探索用の一意なパワー名リストを作る
    all_power_names = []
    for cat_list in POWER_CATEGORIES.values():
        all_power_names.extend(cat_list)
    # 長い順にソート（部分一致防止）
    all_power_names.sort(key=len, reverse=True)

    # パワー名検索
    for pname in all_power_names:
        safe_pname = re.escape(pname)
        # Regex pattern: PNAME + (optional space/colon + optional TYPE) + space + Lv + .? + INT
        pat = rf"{safe_pname}\s*(?:[:：]?\s*([^\s]+))?\s*Lv\.?\s*([0-9oO]+)"

        for m in re.finditer(pat, text):
            # Lv check
            lv_str = m.group(2).translate(str.maketrans("OoLl", "0011"))
            try:
                lv_val = int(lv_str)
            except ValueError:
                continue

            # Type check
            type_val = m.group(1)

            clean_type = None
            if type_val:
                # Type候補が "Lv" や 数字 などのゴミでないか簡易チェック
                if "Lv" in type_val or type_val.isdigit():
                    clean_type = None
                else:
                    clean_type = type_val.strip()

            category = POWER_TO_CATEGORY.get(pname, "unknown")

            extracted_powers.append({
                "name": pname,
                "type": clean_type,
                "lv": lv_val,
                "category": category,
                "full_str": f"{pname}{': ' + clean_type if clean_type else ''} Lv.{lv_val}"
            })

    # 旧フィールド互換のためのマッピング（CSVで見やすくするためにも残す）
    tool_info = next((p for p in extracted_powers if p["name"] == "どうぐパワー"), None)
    # どっさりパワーはTypeがないはずだが、一応
    dossari_info = next((p for p in extracted_powers if p["name"] == "どっさりパワー"), None)
    hoka_info = next((p for p in extracted_powers if p["name"] == "ほかくパワー"), None)

    # 3. Plus Level & Energy
    plus_lv = _find_int([r"\+Lv\.?\s*([0-9oO]+)", r"プラスレベル\s*\+?Lv\.?\s*([0-9oO]+)"], text)
    energy_kcal = _find_int([r"([0-9]{3,5})\s*kca", r"ハラモチエネルギー\s*([0-9]{3,5})"], text.lower())

    # 4. Donut Name
    donut_name = _find_str(
        [r"((?:たまたま|きらきら|ちびちび|でかでか)[^\s]*(?:ミックス|コンフィ|ドーナツ))"],
        text
    )

    return {
        "filename": None,
        "donut_name": donut_name,
        "star_rank": star,
        "plus_lv": plus_lv,
        "energy_kcal": energy_kcal,
        "powers": extracted_powers, # 新構造
        # Legacy / specific helper columns
        "tool_power_type": tool_info["type"] if tool_info else None,
        "tool_power_lv": tool_info["lv"] if tool_info else None,
        "dossari_lv": dossari_info["lv"] if dossari_info else None,
        "capture_type": hoka_info["type"] if hoka_info else None,
        "capture_lv": hoka_info["lv"] if hoka_info else None,
        "raw_text": text,
    }


# -----------------------------
# 集計
# -----------------------------
def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    カテゴリ別集計（確率付き）
    sweet, spicy, sour, bitter, fresh ごとに集計する
    """
    rows = []
    total_count = len(df)
    rows.append(("count", total_count, "100%"))

    if total_count == 0:
        return pd.DataFrame(rows, columns=["metric", "value", "probability"])

    # カテゴリ順に集計
    target_cats = ["sweet", "spicy", "sour", "bitter", "fresh"]

    for cat in target_cats:
        rows.append((f"--- {cat.upper()} ---", "", ""))

        items = []
        for _, row in df.iterrows():
            powers = row.get("powers", [])
            if not isinstance(powers, list):
                continue

            # この行でこのカテゴリに該当するものを探す
            found_strs = []
            for p in powers:
                if p["category"] == cat:
                    found_strs.append(p["full_str"])

            if found_strs:
                items.extend(found_strs)
            else:
                pass

        # 集計
        if items:
            # 頻度順(value_counts default)ではなく、名前順(sort_index)にする
            vc = pd.Series(items).value_counts(sort=False)
            vc = vc.sort_index()

            for k, v in vc.items():
                pct = (v / total_count) * 100
                rows.append((k, int(v), f"{pct:.1f}%"))
        else:
             rows.append(("(None)", 0, "0.0%"))

    # ドーナツ名
    rows.append(("--- DONUT NAME ---", "", ""))
    if "donut_name" in df.columns:
        vc = df["donut_name"].value_counts(dropna=False).sort_index()
        for k, v in vc.items():
            pct = (v / total_count) * 100
            rows.append((f"name={k}", int(v), f"{pct:.1f}%"))

    # エネルギー
    if "energy_kcal" in df.columns:
        rows.append(("--- ENERGY ---", "", ""))
        s = df["energy_kcal"].dropna()
        if len(s) > 0:
            rows.append(("mean", f"{s.mean():.1f}", "-"))
            rows.append(("min", int(s.min()), "-"))
            rows.append(("max", int(s.max()), "-"))

    # --- ITEM + DOSSARI COMBINATION ---
    rows.append(("--- ITEM + DOSSARI COMBINATION ---", "", ""))
    if "tool_power_type" in df.columns and "tool_power_lv" in df.columns and "dossari_lv" in df.columns:
        combinations = []
        for _, row in df.iterrows():
            t_type = row["tool_power_type"]
            t_lv = row["tool_power_lv"]
            d_lv = row["dossari_lv"]

            # Format: "どうぐ{Type}Lv{Lv} + どっさりLv{Lv}"
            # If missing, use "-"
            t_part = f"どうぐ:{t_type} Lv.{int(t_lv)}" if pd.notna(t_type) and pd.notna(t_lv) else "(None)"
            d_part = f"どっさり Lv.{int(d_lv)}" if pd.notna(d_lv) else "(None)"

            combinations.append(f"{t_part} + {d_part}")

        vc = pd.Series(combinations).value_counts(sort=False).sort_index()
        for k, v in vc.items():
            pct = (v / total_count) * 100
            rows.append((k, int(v), f"{pct:.1f}%"))

    # --- SWEET POWER COMBINATIONS ---
    rows.append(("--- SWEET POWER COMBINATIONS ---", "", ""))
    if "powers" in df.columns:
        combinations = []
        for _, row in df.iterrows():
            powers = row.get("powers", [])
            if not isinstance(powers, list):
                combinations.append("(None)")
                continue

            # Filter for sweet category
            sweet_powers = [p for p in powers if p.get("category") == "sweet"]

            if not sweet_powers:
                combinations.append("(None)")
            else:
                # 優先度順にソート (かがやきパワーを先頭に)
                priority = {"かがやきパワー": 0, "オヤブンパワー": 1, "でかでかパワー": 2, "ちびちびパワー": 3}
                sweet_powers.sort(key=lambda x: (priority.get(x.get("name", ""), 99), x.get("name", ""), x.get("lv", 0)))
                # Format: "PowerA Lv.X + PowerB Lv.Y"
                combo_str = " + ".join([p.get("full_str", "") for p in sweet_powers])
                combinations.append(combo_str)

        vc = pd.Series(combinations).value_counts(sort=False).sort_index()
        for k, v in vc.items():
            pct = (v / total_count) * 100
            rows.append((k, int(v), f"{pct:.1f}%"))

    return pd.DataFrame(rows, columns=["metric", "value", "probability"])


# -----------------------------
# メイン
# -----------------------------
def iter_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser(
        description="ドーナツ画像のOCR集計ツール（GPU・並列処理対応）"
    )
    ap.add_argument("--input", required=True, help="Input directory or image file")

    # GPU関連オプション
    ap.add_argument(
        "--gpu",
        choices=["auto", "cuda", "mps", "rocm", "cpu"],
        default="auto",
        help="GPU使用モード (デフォルト: auto)"
    )
    ap.add_argument(
        "--engine",
        choices=["easyocr"],
        default=None,
        help="OCRエンジンを強制指定（現在はeasyocrのみ対応）"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="ワーカー数（デフォルト: 自動）"
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="ベンチマークモード（最初の10枚で最適設定を探す）"
    )

    # 出力オプション
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_out = f"extracted_{now_str}.csv"
    default_summary = f"summary_{now_str}.csv"

    ap.add_argument("--out", default=default_out, help=f"Output CSV filename (default: {default_out})")
    ap.add_argument("--summary", default=default_summary, help=f"Summary CSV filename (default: {default_summary})")
    ap.add_argument("--output-dir", default=".", help="Output directory path")

    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # 出力ディレクトリの準備
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 出力パスの構築
    out_csv_path = output_dir / args.out
    summary_csv_path = output_dir / args.summary

    image_paths = iter_images(input_path)
    print(f"Found {len(image_paths)} images in {input_path}")

    if not image_paths:
        print("No images found.")
        return

    # GPU検出
    gpu_type, gpu_info = GPUDetector.detect()
    print(f"Detected GPU: {gpu_type.value} ({gpu_info})")

    # GPUモードの決定
    if args.gpu == "auto":
        use_gpu = gpu_type != GPUType.CPU
    else:
        use_gpu = args.gpu != "cpu"
        # 明示的にGPUタイプを上書き
        if args.gpu == "cuda":
            gpu_type = GPUType.NVIDIA_CUDA
        elif args.gpu == "mps":
            gpu_type = GPUType.APPLE_MPS
        elif args.gpu == "rocm":
            gpu_type = GPUType.AMD_ROCM
        elif args.gpu == "cpu":
            gpu_type = GPUType.CPU

    # OCRエンジンの作成
    ocr_engine = create_ocr_engine(
        gpu_type=gpu_type,
        force_engine=args.engine,
        use_gpu=use_gpu
    )
    print(f"OCR Engine: {ocr_engine.__class__.__name__}")
    print(f"Use GPU: {use_gpu}")

    # 並列OCRエンジンの作成
    num_workers = calculate_optimal_workers(
        use_gpu=use_gpu,
        total_images=len(image_paths)
    )
    if args.workers:
        num_workers = args.workers

    print(f"Workers: {num_workers}")
    print()

    # ベンチマークモード
    if args.benchmark:
        from parallel_ocr import benchmark_processing
        print("=== ベンチマークモード ===")
        benchmark_processing(ocr_engine, image_paths)
        return

    # 並列OCR処理
    parallel_engine = ParallelOCREngine(
        ocr_engine=ocr_engine,
        num_workers=num_workers,
        use_threading=use_gpu  # GPU使用時はスレッドプール推奨
    )

    print("=== OCR処理開始 ===")
    results = parallel_engine.process_images(image_paths)

    # 結果をパース
    parsed_results = []
    errors = 0
    for img_path, text in results:
        if text.startswith("ERROR:"):
            errors += 1
            # エラーでも空のデータを作成
            data = parse_fields("")
            data["filename"] = img_path.name
            data["raw_text"] = text
        else:
            data = parse_fields(text)
            data["filename"] = img_path.name
            data["raw_text"] = text

        parsed_results.append(data)

    if errors > 0:
        print(f"\n⚠️  {errors}件の処理エラーがありました")

    if not parsed_results:
        print("No data extracted.")
        return

    df = pd.DataFrame(parsed_results)

    # CSV出力用に powers (list) を文字列化するカラムを作っておくと便利
    def format_powers(powers_list):
        if not isinstance(powers_list, list): return ""
        # cat: full_str list
        cat_map = {c: [] for c in POWER_CATEGORIES.keys()}
        # unknown
        cat_map["unknown"] = []

        for p in powers_list:
            c = p.get("category", "unknown")
            if c in cat_map:
                cat_map[c].append(p.get("full_str", ""))
            else:
                cat_map["unknown"].append(p.get("full_str", ""))

        # 文字列化: "sweet=[...], spicy=[...]"
        parts = []
        for c, plist in cat_map.items():
            if plist:
                parts.append(f"{c}=[{', '.join(plist)}]")
        return " / ".join(parts)

    if "powers" in df.columns:
        df["powers_summary"] = df["powers"].apply(format_powers)

    # カラム順序を整頓
    cols = [
        "filename", "donut_name", "star_rank",
        "plus_lv", "energy_kcal",
        "powers_summary", # 新カラム
        "tool_power_type", "tool_power_lv", # 旧互換
        "dossari_lv",
        "capture_type", "capture_lv",
        "raw_text"
    ]
    # 実際にあるカラムだけ残す
    final_cols = [c for c in cols if c in df.columns]

    df_out = df[final_cols]

    print(f"\nSave extraction result to {out_csv_path}")
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

    # 集計 (元の df を使う。powers listが必要なので)
    df_summary = make_summary(df)
    print(f"Save summary to {summary_csv_path}")
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    # 簡易表示
    print("-" * 40)
    print(df_summary.to_string(index=False))
    print("-" * 40)


if __name__ == "__main__":
    main()
