import pickle
import argparse
import os
import sys
from itertools import chain
import utils
import pandas as pd
import json

def config2dict(s):
    return { item.split("=")[0]: item.split("=")[1] for item in s.split(",") } 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)
    parser.add_argument("--first_k_iter", type=int, default=None)
    parser.add_argument("--contain_config_str", type=str, default=None)

    args = parser.parse_args()

    logdirs = utils.parse_logdirs(args.logdirs)

    if args.key_stats is None:
        key_stats = []
    else:
        key_stats = args.key_stats.split(",")

    if args.contain_config_str is not None: 
        config_strs = config2dict(args.contain_config_str)
    else:
        config_strs = None

    for logdir in logdirs:
        print(f"Processing {logdir}")
        summary_dir = utils.get_checkpoint_summary_path()
        prefix = os.path.join(summary_dir, logdir.replace("/", "_"))

        filename = prefix + ".pkl"
        df = pickle.load(open(filename, "rb"))["df"]

        res = []
        for col in df.columns:
            if col not in key_stats:
                continue
            len_series = df[col].apply(lambda x: len(x) if isinstance(x, list) else 1)

            data = []
            for row_idx, v in enumerate(df[col].values):
                if isinstance(v, float):
                    v = [v]

                for sample_idx, vv in enumerate(v): 
                    if args.first_k_iter is not None and sample_idx > args.first_k_iter:
                        continue

                    if config_strs is not None:
                        skip = False
                        config_strs_row = config2dict(df["_config_str"][row_idx])

                        for k, v in config_strs.items():
                            if config_strs_row.get(k, None) != v:
                                skip = True
                                break
                        if skip:
                            continue

                    data.append((vv, df["folder"][row_idx], df["_config_str"][row_idx], sample_idx))

            data = sorted(data, key = lambda x: x[0])
            mean = sum([ v[0] for v in data ]) / len(data)

            entry = dict(
                key = col,
                min = data[0][0],
                max = data[-1][0],
                mean = mean,
                min_len = min(len_series),
                max_len = max(len_series),
                mean_len = sum(len_series) / len(len_series)
            )

            print(f"Top 10 of {col}")
            for i in range(10):
                print(f"{data[-i-1]}")

            json_filename = prefix + "_top.json" 
            json.dump(data[-10:], open(json_filename, "w")) 

            print(f"Save json to {json_filename}")

            res.append(entry)

        df_stats = pd.DataFrame(res)
        print(df_stats)


if __name__ == "__main__":
    main()
