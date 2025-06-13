import pandas as pd
from sklearn.model_selection import train_test_split
import os


csv_path = "data.csv"
out_dir = "input"
os.makedirs(out_dir, exist_ok=True)


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(out_dir, csv_path))

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    valid_df.to_csv(os.path.join(out_dir, "valid.csv"), index=False)

    print(f"Train size: {len(train_df)}")
    print(f"Valid size: {len(valid_df)}")
