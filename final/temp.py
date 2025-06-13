import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV 파일 경로")
    args = parser.parse_args()

    # CSV 파일 로드
    df = pd.read_csv(args.csv)
    
    col_list = ['bleu','rouge_l','meteor','bertscore']

    # 지정한 열만 추출
    if not all(col in df.columns for col in col_list):
        missing = [col for col in col_list if col not in df.columns]
        raise ValueError(f"다음 열이 존재하지 않습니다: {missing}")

    df_selected = df[col_list]

    # 덮어쓰기
    df_selected.to_csv(args.csv, index=False)

    # 각 열의 평균 출력
    print("각 열의 평균:")
    for col in col_list:
        mean_val = df_selected[col].mean()
        print(f"{col}: {mean_val:.4f}")

if __name__ == "__main__":
    main()
