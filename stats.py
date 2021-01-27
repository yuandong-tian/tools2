import pickle
import argparse
import os
import sys
import numpy as np
import math
import utils
import pandas as pd
import json
from tabulate import tabulate
from collections import OrderedDict

from utils import signature

from utils_stats import *

def print_top_n(col, df, args):
    num_rows = df.shape[0]
    n = min(args.topk, num_rows)
    df_sorted = df.sort_values(by=[col], ascending=not args.descending)

    print(f"Top {n}/{num_rows} of {col} (each row takes average of top-{args.topk_mean} entries):")
    print(tabulate(df_sorted.head(n), headers='keys'))

def print_col_infos(df):
    # Print out possible values in the columns. 
    cond_vars = {}
    sweep_vars = {}
    override_prefix = "override_" 
    for name, column in df.iteritems():
        if name.startswith(override_prefix):
            rec = column.unique()
            s = { name[len(override_prefix):] : rec }
            if len(rec) == 1:
                cond_vars.update(s)
            else:
                sweep_vars.update(s)
    print()
    print("Conditional variables: ")
    for k, v in cond_vars.items():
        print(f"{k}: {v}")
    print()
    print("Sweep variables:")
    for k, v in sweep_vars.items():
        print(f"{k}: {v}")
    print()

    return cond_vars,sweep_vars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)
    parser.add_argument("--subkey", type=str, default=None, help="subkey needs to be specified if the time series is a list of dict")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--first_k_iter", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--groups", type=str, default=None,
                        help="comma separated 'key=type' string to convert columns to different types than string. if set to '/', then groups automatically to sweep parameters")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--topk_mean", type=int, default=5)
    parser.add_argument("--output_no_save", action="store_true")

    command_line = " ".join(sys.argv)

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
    if args.groups is not None and args.groups != "/":
        groups = OrderedDict()
        for kv in args.groups.split(","):
            items = kv.split("=")
            groups[items[0]] = items[1] if len(items) > 1 else 'str'
    
    summary_dir = utils.get_checkpoint_summary_path()

    res = []
    filename = "stats_" + signature() + ".txt"
    default_stdout = sys.stdout 
    
    for logdir in logdirs:
        prefix = os.path.join(summary_dir, logdir.replace("/", "_"))

        if not args.output_no_save:
            if not os.path.exists(prefix):
                os.mkdir(prefix)
            log_record = os.path.join(prefix, filename)
            print(f"Output stored in {log_record}")

            sys.stdout = open(log_record, "w") 
            print(command_line)

        print(f"Processing {logdir}")
        filename = prefix + ".pkl"
        df = pickle.load(open(filename, "rb"))["df"]

        # keep those records that satisfy config_filter
        sel = df.apply(config_filter, axis=1, args=(config_strs,))
        df = df[sel]
        if df.shape[0] == 0:
            print("No selection!")
            continue

        # Process the records. 
        df = df.apply(process_func, axis=1, args=(key_stats, args))

        # Print information in each column 
        cond_vars, sweep_vars = print_col_infos(df)
        if args.groups == "/":
            groups = OrderedDict()
            for k, v in sweep_vars.items():
                if k != "seed":
                    groups[k] = 'str'

        print("Recent modified since: ", df["modified_since"].min())

        for col in key_stats:
            sel = [col, "folder", "modified_since", "_config_str", f"{col}_best_idx", f"{col}_len"]
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
            c = df[col]["mean"]
            print(f"max_val: {c.max()} at " + ",".join([f"{g}={v}" for g, v in zip(groups, c.idxmax())]))
            print(f"min_val: {c.min()} at " + ",".join([f"{g}={v}" for g, v in zip(groups, c.idxmin())]))

        # json_filename = prefix + "_top.json" 
        # json.dump(res, open(json_filename, "w")) 
        # print(f"Save json to {json_filename}")
        if not args.output_no_save:
            f = sys.stdout
            sys.stdout = default_stdout 
            f.close()
            with open(log_record, "r") as f:
                for line in f:
                    print(line, end='')

    df_stats = pd.DataFrame(res)
    print(df_stats)

if __name__ == "__main__":
    main()
