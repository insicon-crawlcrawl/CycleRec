import os
import glob
import pandas as pd
import codecs
import chardet
import traceback

def detect_encoding(file_path, nbytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(nbytes)
    return chardet.detect(raw)['encoding'] or 'utf-8'

def extract_empty_asins(input_dir, output_path):
    # Find all CSV files in the input directory
    pattern = os.path.join(input_dir, '*.csv')
    files = glob.glob(pattern)
    if not files:
        print(f"No CSV files found in {input_dir}.")
        return

    empty_asins = set()
    for f in files:
        enc = detect_encoding(f)
        try:
            # codecs.open으로 errors 처리
            with codecs.open(f, 'r', encoding=enc, errors='replace') as reader:
                df = pd.read_csv(reader)
        except Exception as e:
            print("----- CSV 읽기 실패 디버깅 정보 -----")
            print(f"파일 경로: {f}")
            print(f"추정 인코딩: {enc}")
            print(f"예외 메시지: {e}")
            traceback.print_exc()
            print("-----------------------------------")
            continue

        if 'ASIN' not in df.columns:
            continue

        # Identify columns to check: all except ASIN and missing_count (if present)
        other_cols = [c for c in df.columns if c not in ('ASIN', 'missing_count')]
        # Rows where all other columns are NaN
        mask = df[other_cols].isna().all(axis=1)
        # Add ASINs of those rows to set
        empty_asins.update(df.loc[mask, 'ASIN'].dropna().astype(str).tolist())

    if not empty_asins:
        print("추출된 empty ASIN이 없습니다.")
        return

    # Convert to sorted list and save
    sorted_asins = sorted(empty_asins)
    out_df = pd.DataFrame({'ASIN': sorted_asins})
    out_df.to_csv(output_path, index=False)
    print(f"Extracted {len(sorted_asins)} empty ASINs to {output_path}.")

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, 'output-ver4')
    output_file = os.path.join(base_dir, 'empty_asins_ver4.csv')
    extract_empty_asins(input_dir, output_file)
