import pickle
import argparse
import os
import sys
from itertools import chain
import utils
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)

    args = parser.parse_args()

    logdirs = utils.parse_logdirs(args.logdirs)

    if args.key_stats is None:
        key_stats = []
    else:
        key_stats = args.key_stats.split(",")


    for logdir in logdirs:
        print(f"Processing {logdir}")
        filename = os.path.join(utils.get_checkpoint_summary_path(), logdir.replace("/", "_") + ".pkl")
        df = pickle.load(open(filename, "rb"))["df"]

        res = []
        for col in df.columns:
            if col not in key_stats:
                continue
            len_series = df[col].apply(lambda x: len(x))
            data = list(chain.from_iterable(df[col].values)) 

            entry = dict(
                key = col,
                min = min(data),
                max = max(data),
                mean = sum(data) / len(data),
                min_len = min(len_series),
                max_len = max(len_series),
                mean_len = sum(len_series) / len(len_series)
            )

            data = [ (vv, df["folder"][row_idx], sample_idx) for row_idx, v in enumerate(df[col].values) for sample_idx, vv in enumerate(v) ] 
            data = sorted(data, key = lambda x: x[0])
            print(f"Top 10 of {col}")
            for i in range(10):
                print(f"{data[-i-1]}")

            res.append(entry)

        df_stats = pd.DataFrame(res)
        print(df_stats)


if __name__ == "__main__":
    main()
