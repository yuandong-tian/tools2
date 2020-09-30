import pickle
import argparse
import os
import sys
from itertools import chain
import numpy as np
import math
import utils
import pandas as pd
import json
from tabulate import tabulate
from collections import OrderedDict

def config2dict(s):
    return { item.split("=")[0]: item.split("=")[1] for item in s.split(",") } 

def print_top_n(col, df, args):
    num_rows = df.shape[0]
    n = min(args.topk, num_rows)
    df_sorted = df.sort_values(by=[col], ascending=not args.descending)

    print(f"Top {n}/{num_rows} of {col} (each row takes average of top-{args.topk_mean} entries):")
    print(tabulate(df_sorted.head(n), headers='keys'))

def config_filter(row, config_strs):
    if config_strs is None:
        return True

    config = config2dict(row["_config_str"])

    for k, v in config_strs.items():
        if config.get(k, None) != v:
            return False
    return True

def group_func(row, groups):
    config = config2dict(row["_config_str"])
    for k in groups:
        row[k] = config.get(k, None)
    del row["_config_str"]
    return row

def process_func(row, cols, args):
    if cols is None:
        return row

    # For config str, remove 'githash' and 'sweep_filename'
    config = config2dict(row["_config_str"])
    row["_config_str"] = ",".join([f"{k}={v}" for k, v in config.items() if not k in ('githash', 'sweep_filename')])

    for col in cols:
        data = row[col]
        if args.first_k_iter is not None:
            data = row[col][:args.first_k_iter]

        if len(data) > 0:
            data = np.array(data) 
            inds = data.argsort()
            if args.descending:
                inds = inds[::-1]
            data = data[inds]

            best = data[0]
            best_idx = inds[0]

            if len(data) < args.topk_mean:
                topk_mean = sum(data) / len(data)
            else:
                topk_mean = sum(data[:args.topk_mean]) / args.topk_mean
        else:
            best = None
            best_idx = None
            topk_mean = None

        row[col] = topk_mean
        row[f"{col}_best"] = best
        row[f"{col}_best_idx"] = best_idx
        row[f"{col}_len"] = len(data)

    return row 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--first_k_iter", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--groups", type=str, default=None,
                        help="comma separated 'key=type' string to convert columns to different types than string")
    parser.add_argument("--topk", type=int, default=10)
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

    groups = None
    if args.groups is not None:
        groups = OrderedDict()
        for kv in args.groups.split(","):
            items = kv.split("=")
            groups[items[0]] = items[1] if len(items) > 1 else 'str'
    
    res = []
    
    for logdir in logdirs:
        print(f"Processing {logdir}")
        summary_dir = utils.get_checkpoint_summary_path()
        prefix = os.path.join(summary_dir, logdir.replace("/", "_"))

        filename = prefix + ".pkl"
        df = pickle.load(open(filename, "rb"))["df"]

        sel = df.apply(config_filter, axis=1, args=(config_strs,))
        df = df[sel]
        df = df.apply(process_func, axis=1, args=(key_stats, args))

        for col in key_stats:
            sel = [col, "folder", "_config_str", f"{col}_best_idx", f"{col}_len"]
            data = df[sel]
            print_top_n(col, data, args)

            d = df[col]
            entry = dict(
                logdir=logdir,
                key=col,
                min=np.min(d),
                max=np.max(d),
                mean=np.mean(d),
                std=np.std(d),
            )
            res.append(entry)

        # Print out group means.
        if groups is not None:
            cols = [ col for col in key_stats ] + [ "_config_str" ]
            aggs = { col: [ 'mean', 'std' ] for col in key_stats }
            df = df[cols].apply(group_func, axis=1, args=(groups,))
            df = df.astype(groups)
            df = df.groupby(list(groups.keys())).agg(aggs)
            print(df)

        # json_filename = prefix + "_top.json" 
        # json.dump(res, open(json_filename, "w")) 
        # print(f"Save json to {json_filename}")

    df_stats = pd.DataFrame(res)
    print(df_stats)

if __name__ == "__main__":
    main()
