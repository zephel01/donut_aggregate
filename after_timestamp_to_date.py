import os
import re
import shutil
import argparse
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Filename timestamp to file date (with copy)")
    parser.get_default = lambda x: None
    parser.add_argument("--input", type=str, default=".", help="Source directory")
    parser.add_argument("--output", type=str, default="date_fixed", help="Destination directory")
    args = parser.parse_args()

    # 対象フォルダ
    src_folder = Path(args.input)
    dst_folder = Path(args.output)

    if not src_folder.exists():
        print(f"入力フォルダが見つかりません: {src_folder}")
        return

    # 出力フォルダ作成
    dst_folder.mkdir(parents=True, exist_ok=True)

    # サポートする拡張子
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    
    # 全ファイルを再帰的に探索
    image_paths = sorted([p for p in src_folder.rglob("*") if p.suffix.lower() in exts])
    print(f"Found {len(image_paths)} images in {src_folder}")

    count = 0
    skipped = 0
    skipped_reasons = {} # Reason -> list of examples

    print(f"Starting processing for {len(image_paths)} files...")

    for src_path in image_paths:
        filename = src_path.name
        
        # ファイル名に含まれるすべての連続する数字を抽出
        all_numbers = re.findall(r'\d+', filename)
        
        best_ts = None
        best_time = None
        skip_reason = "No 9-13 digit number found"
        
        for num_str in all_numbers:
            # 9桁から13桁の数字を候補とする (秒 または ミリ秒)
            if 9 <= len(num_str) <= 13:
                try:
                    ts = int(num_str)
                    # ミリ秒(13桁)なら秒に変換
                    if len(num_str) >= 12:
                        ts = ts // 1000
                    
                    # 極端に古い/未来のタイムスタンプは除外 (2000年〜2050年)
                    if 946684800 <= ts <= 2524608000:
                        best_ts = ts
                        best_time = datetime.fromtimestamp(ts)
                        break # 最初に見つかった妥当な数字を採用
                    else:
                        skip_reason = f"Timestamp out of range: {ts}"
                except:
                    skip_reason = f"Invalid timestamp format: {num_str}"
                    continue
        
        if not best_ts:
            reason_key = skip_reason
            if reason_key not in skipped_reasons:
                skipped_reasons[reason_key] = []
            if len(skipped_reasons[reason_key]) < 5:
                skipped_reasons[reason_key].append(filename)
            skipped += 1
            continue
        
        # 日付文字列 (YYYY-MM-DD) を作成してサブフォルダを決定
        date_str = best_time.strftime('%Y-%m-%d')
        target_subfolder = dst_folder / date_str
        target_subfolder.mkdir(parents=True, exist_ok=True)
        
        dst_path = target_subfolder / filename
        
        # 指定したフォルダにコピー (メタデータ保持)
        try:
            shutil.copy2(src_path, dst_path)
            # コピー先のファイルの作成日時と更新日時を変更
            os.utime(dst_path, (best_ts, best_ts))   # (アクセス時刻, 更新時刻)
            count += 1
            if count % 500 == 0:
                print(f"Processed {count} files...")
        except Exception as e:
            reason_key = f"System Error: {str(e)}"
            if reason_key not in skipped_reasons:
                skipped_reasons[reason_key] = []
            skipped_reasons[reason_key].append(filename)
            skipped += 1

    print("-" * 40)
    print(f"完了: {count} 個のファイルを整理しました。")
    if skipped > 0:
        print(f"スキップ: {skipped} 個のファイルが処理されませんでした。")
        print("\nスキップの主な理由と例:")
        for reason, examples in skipped_reasons.items():
            print(f"  [{reason}]")
            for ex in examples[:3]:
                print(f"    - {ex}")

if __name__ == "__main__":
    main()