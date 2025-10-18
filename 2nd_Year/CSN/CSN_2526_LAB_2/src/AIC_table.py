import pandas as pd
import numpy as np
import os

def AIC(logL, K, N):
    if N-K-1<=0:
        return np.nan
    return -2*logL + 2*K*(N/(N-K-1))

def process_file(file_path,N):
    df = pd.read_csv(file_path)
    df['K'] = df['Parameters'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and str(x) != '[]' else 0)
    df['AICc'] = df.apply(lambda row: AIC(row['Log_Likelihood'], row['K'], N), axis=1)
    best_AICc = df['AICc'].min()
    df["Delta_AICc"] = df['AICc'] - best_AICc

    return df

def main():
    input_dir = "../data/lang_log_like"
    output_dir = "../data/lang_AIC"
    degree_dir = "../data/degree_sequences"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        print(f"Processing {file}")

        language = os.path.splitext(file)[0].split('_')[0]
        degree_file = os.path.join(degree_dir, f"{language}_degree_sequence.txt")
        degree_sequence = np.loadtxt(degree_file)
        N = len(degree_sequence[degree_sequence > 0])

        df = process_file(file_path, N)

        out_path = os.path.join(output_dir, file)
        df.to_csv(out_path, index=False)
        print(f"Saved processed data to {out_path}")

if __name__ == "__main__":
    main()