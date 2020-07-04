import pickle
import argparse
import os
import sys
from itertools import chain
import math
import utils
import pandas as pd
import json

def config2dict(s):
    return { item.split("=")[0]: item.split("=")[1] for item in s.split(",") } 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--first_k_iter", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--topk_mean", type=int, default=5)

    args = parser.parse_args()

    logdirs = utils.parse_logdirs(args.logdirs)

    if args.key_stats is None:
        key_stats = []
    else:
        key_stats = args.key_stats.split(",")

    if args.config is not None: 
        config_strs = config2dict(args.config)
    else:
        config_strs = None
    
    order_func = lambda x: -x[0] if args.descending else x[0]

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

                this_data = []
                for sample_idx, vv in enumerate(v): 
                    if args.first_k_iter is not None and sample_idx > args.first_k_iter:
                        continue

                    if math.isnan(vv):
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

                    this_data.append((vv, sample_idx))

                # Compute top-5 average within a row. 
                if len(this_data) == 0:
                    continue

                this_data = sorted(this_data, key=order_func)
                best = this_data[0][1]
                this_data = [ vv for vv, idx in this_data]

                if len(this_data) < args.topk_mean:
                    avg = sum(this_data) / len(this_data)
                else:
                    avg = sum(this_data[:args.topk_mean]) / args.topk_mean

                data.append((avg, df["folder"][row_idx], df["_config_str"][row_idx], best, len(this_data)))

            data = sorted(data, key=order_func)
            mean = sum([ v[0] for v in data ]) / len(data)

            entry = dict(
                key = col,
                min = data[-1][0],
                max = data[0][0],
                mean = mean,
                min_len = min(len_series),
                max_len = max(len_series),
                mean_len = sum(len_series) / len(len_series)
            )

            n = min(10, len(data))

            print(f"Top {n} of {col} (each row takes average of top-{args.topk_mean} entries):")
            for i in range(n):
                print(f"{data[i]}")

            json_filename = prefix + "_top.json" 
            json.dump(data[:n], open(json_filename, "w")) 

            print(f"Save json to {json_filename}")

            res.append(entry)

        df_stats = pd.DataFrame(res)
        print(df_stats)


if __name__ == "__main__":
    main()
