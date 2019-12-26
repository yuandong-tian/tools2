import pickle
import argparse
import os
import sys
from itertools import chain
import pandas as pd

def main():
    checkpoint_path = f"/checkpoint/{os.environ['USER']}" 

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--key_stats", type=str, default=None)

    args = parser.parse_args()

    filename = os.path.join(checkpoint_path, "summary", args.logdir.replace("/", "_") + ".pkl")
    df = pickle.load(open(filename, "rb"))["df"]

    if args.key_stats is None:
        key_stats = []
    else:
        key_stats = args.key_stats.split(",")

    res = []
    for col in df.columns:
        if col not in key_stats:
            continue
        len_series = df[col].apply(lambda x: len(x))

        entry = dict(
            key = col,
            min = min(chain.from_iterable(df[col].values)),
            max = max(chain.from_iterable(df[col].values)),
            min_len = min(len_series),
            max_len = max(len_series),
            mean_len = sum(len_series) / len(len_series)
        )

        res.append(entry)

    df_stats = pd.DataFrame(res)
    print(df_stats)


if __name__ == "__main__":
    main()
