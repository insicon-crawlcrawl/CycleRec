import os
import glob
import pandas as pd
import numpy as np
import chardet
import codecs
import traceback

def detect_encoding(file_path, nbytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(nbytes)
    return chardet.detect(raw)['encoding'] or 'utf-8'

def integrate_csvs(input_dir, output_path):
    # Expected columns
    cols = [
        'ASIN', 'Title', 'Rating', 'Rating Count', 'Bought Last Month',
        'Category 1', 'Category 2', 'Category 3', 'Category 4',
        'Category 5', 'Category 6', 'Category 7', 'Category 8'
    ]

    # Locate all CSV files in input directory
    pattern = os.path.join(input_dir, '*.csv')
    files = glob.glob(pattern)
    if not files:
        print(f"No CSV files found in {input_dir}.")
        return

    dfs = []
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
            print("전체 스택트레이스:")
            traceback.print_exc()
            print("-----------------------------------")
            continue

        # Ensure all expected columns are present
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        dfs.append(df[cols])

    if not dfs:
        print("No dataframes were read successfully; exiting.")
        return

    # Concatenate all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # Drop rows where all columns except ASIN are NaN
    other_cols = [c for c in cols if c != 'ASIN']
    combined = combined.dropna(subset=other_cols, how='all')

    # For each ASIN, keep row with the most non-null values
    combined['data_count'] = combined[other_cols].notna().sum(axis=1)
    combined = combined.sort_values(['ASIN', 'data_count'], ascending=[True, False])
    combined = combined.drop_duplicates(subset='ASIN', keep='first')

    # Final sorting by ASIN and cleanup
    combined = combined.sort_values('ASIN').drop(columns=['data_count'])

    # Save the integrated CSV
    combined.to_csv(output_path, index=False)
    print(f"Integrated CSV written to {output_path}.")

if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, 'output', 'output-ver4')
    output_file = os.path.join(base_dir, '..', 'data', 'amazon-web-category-ver4.csv')
    integrate_csvs(input_dir, output_file)
