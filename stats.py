import pickle
import argparse
import os
import sys
import re
import numpy as np
import math
import utils
import pandas as pd
import json
from tabulate import tabulate
import itertools
from collections import OrderedDict

from utils import signature

from utils_stats import *
from common_utils import MultiRunUtil

pd.options.display.max_rows = 1000
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def print_top_n(col, df, params_getter):
    num_rows = df.shape[0]
    n = min(params_getter(col, "topk"), num_rows)
    df_sorted = df.sort_values(by=[col], ascending=not params_getter(col, "descending"))

    print(f"Top {n}/{num_rows} of {col} (each row takes average of top-{params_getter(col, 'topk_mean')} entries):")
    print(tabulate(df_sorted.head(n), headers='keys'))

def print_col_infos(df, additional_sweep_vars=[]):
    # Print out possible values in the columns. 
    cond_vars = {}
    sweep_vars = {}
    override_prefix = "override_" 
    for name, column in df.iteritems():
        if name.startswith(override_prefix):
            rec = sorted(column.unique())
            s = { name[len(override_prefix):] : rec }
            if len(rec) == 1:
                cond_vars.update(s)
            else:
                sweep_vars.update(s)
        elif name in additional_sweep_vars:
            rec = sorted(column.unique())
            sweep_vars[name] = rec 
            
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
    

matcher = re.compile(r"([^,= ]+)=(\[.*?\])|([^,= ]+)")

def get_var_range(vars_str, sweep_vars, types_convert=None):
    names = []
    choices = []

    for m in matcher.finditer(vars_str): 
        name = m.group(1) or m.group(3)
        v = m.group(2)
        assert name is not None, f"name cannot be None. {vars_str}"

        if v is not None:
            names.append(name)
            choices.append(eval(v))
        else:
            names.append(name)
            choices.append(sweep_vars[name])

    return names, choices

def print_latex(df, row_names, row_choices, col_names, col_choices, key_stat, precision=2):
    s = ""
    for col_entry in itertools.product(*col_choices):
        col_name = ",".join([ f"{e}" for v, e in zip(col_names, col_entry) ]) 
        s += f"& {col_name} "
    s += "\\\\ \n"

    for row_entry in itertools.product(*row_choices):
        row_name = ",".join([ f"{e}" for v, e in zip(row_names, row_entry) ]) 

        cond_row = True
        for v, e in zip(row_names, row_entry):
            cond_row = cond_row & (df["override_" + v] == e)

        s += f"{row_name}"

        for col_entry in itertools.product(*col_choices):
            cond_col = True
            for v, e in zip(col_names, col_entry):
                cond_col = cond_col & (df["override_" + v] == e)

            sel_value = df[cond_row & cond_col][key_stat]
            mean_v = sel_value.mean()
            std_v = sel_value.std()
            s += f"& ${mean_v:.{precision}f}\pm {std_v:.{precision}f}$"

        s += "\\\\ \n"

    return s

def get_type_spec(type_str):
    types = dict()
    if type_str is not None:
        for kv in type_str.split(","):
            items = kv.split("=")
            types["override_" + items[0]] = items[1] if len(items) > 1 else 'str'

    return types

def get_group_spec(group_str, sweep_vars):
    if group_str == "/":
        return [ key for key in sweep_vars.keys() if key != "seed"]
    elif group_str is not None:
        return group_str.split(",")
    else:
        return None

class MeanStdAggFunc:
    def __init__(self, precision=2, use_latex=False):
        self.precision = precision
        self.use_latex = use_latex

    def agg(self, series):
        mean_val = series.mean()
        std_val = series.std()
        nona_cnt = series.count() 
        if self.use_latex:
            return fr"${mean_val:.{self.precision}f}\pm {std_val:.{self.precision}f}$"
        else:
            return f"{mean_val:.{self.precision}f} ± {std_val:.{self.precision}f} [{nona_cnt}]"


class FolderAggFunc:
    def __init__(self, max_simple=5):
        self.max_simple = max_simple

    def agg(self, series):
        try:
            ss = sorted([ int(os.path.basename(f)) for f in series ])
        except:
            ss = [0] 

        if len(ss) == 0:
            return "[0]"

        # [6,7,8,10,12,13] -> 6-8,10,12-13
        last_start = ss[0]
        last_end = ss[0]
        output = []
        for s in ss[1:]:
            if s > last_end + 1:
                if last_end > last_start:
                    output.append(f"{last_start}-{last_end}")
                else:
                    output.append(f"{last_start}")
                last_start = last_end = s
            else:
                last_end = s 

        if last_end > last_start:
            output.append(f"{last_start}-{last_end}")
        else:
            output.append(f"{last_start}")

        return ",".join(output) + f"[{series.size}]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", type=str)
    parser.add_argument("--key_stats", type=str, default=None)
    parser.add_argument("--subkey", type=str, default=None, help="subkey needs to be specified if the time series is a list of dict")
    parser.add_argument("--descending", action="store_true")
    parser.add_argument("--first_k_iter", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--groups", type=str, default=None,
                        help="comma separated 'key=type' string to convert columns to different types than string. if set to '/', then groups automatically to sweep parameters")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--topk_mean", type=int, default=5)
    parser.add_argument("--output_no_save", action="store_true")

    parser.add_argument("--types", type=str, default=None, help="Specifying types with comma-separated key=type") 

    parser.add_argument("--rows", type=str, default=None, 
                        help='''
                        Make latex table, specifying rows. 
                            If --rows "A=[1,2],B=[3,4]"
                            then it will have four rows (A=1,B=3), (A=1,B=4), (A=2,B=3), (A=2,B=4)
                            If we just specify --rows "A,B", then A and B will use all the values in the sweep_vars.
                        ''')
    parser.add_argument("--cols", type=str, default=None, help="Make latex table, specifying cols. The specification is similar to rows. For each (row, col) pair, the metric will be averaged to yield mean and std")
    parser.add_argument("--precision", type=int, default=3, help="Precision used in display")
    parser.add_argument("--print_top_n", action="store_true")
    parser.add_argument("--use_latex_agg", action="store_true")
    parser.add_argument("--save_grouped_table", action="store_true")
    parser.add_argument("--additional_sweep_vars", type=str, default="")
    parser.add_argument("--iter_thres", type=str, default="", help="specifying 'iter_thres acc=300' means that a record will be included if its acc entry has more than 300 entries.")

    command_line = " ".join(sys.argv)

    args = parser.parse_args()

    logdirs = utils.parse_logdirs(args.logdirs)

    if args.config is not None: 
        config_strs = config2dict(args.config)
    else:
        config_strs = None

    summary_dir = utils.get_checkpoint_summary_path()

    res = []
    sig = signature() 
    filename = "stats_" + sig + ".txt"
    output_tbl_filename = "tbl_" + sig + ".pkl"
    default_stdout = sys.stdout 

    if args.iter_thres != "":
        iter_thres = { item.split("=")[0] : int(item.split("=")[1]) for item in args.iter_thres.split(",") }
    else:
        iter_thres = {}
    
    for logdir in logdirs:
        logdir = utils.preprocess_logdir(logdir)
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
        data = pickle.load(open(filename, "rb"))

        mdl = MultiRunUtil.load_check_module(data["meta"]["subfolders"][0])
        if args.key_stats is None and hasattr(mdl, "_attr_multirun") and "default_metrics" in mdl._attr_multirun: 
            key_stats = mdl._attr_multirun["default_metrics"]
        else:
            key_stats = args.key_stats.split(",")

        def get_metric_info(metric, param_name):
            if hasattr(mdl, "_attr_multirun"):
                params = mdl._attr_multirun.get("common_options", {})
                # Overwrite common keys. 
                params.update(mdl._attr_multirun["specific_options"].get(metric, {}))
                return params.get(param_name, getattr(args, param_name))
            else:
                return getattr(args, param_name) 

        def reorder_group(groups):
            if hasattr(mdl, "_attr_multirun") and "default_group_order" in mdl._attr_multirun: 
                default_order = mdl._attr_multirun["default_group_order"]
                default_order_in_groups = [ a for a in default_order if a in groups ]
                remaining = [ a for a in groups if a not in default_order ] 
                return default_order_in_groups + remaining
            else:
                return groups
            
        df = data["df"]

        # keep those records that satisfy config_filter
        sel = df.apply(config_filter, axis=1, args=(config_strs,iter_thres))
        df = df[sel]
        if df.shape[0] == 0:
            print("No selection!")
            continue

        # Process the records. 
        df = df.apply(process_func, axis=1, args=(key_stats, get_metric_info, args.first_k_iter))

        types_convert = get_type_spec(args.types)
        if args.first_k_iter is not None:
            # They are used as suffix to key_stats
            iters = args.first_k_iter.split(",")
            key_stats = [ combine_key_iter(col, i) for col in key_stats for i in iters ] 
            types_convert = { combine_key_iter(col, i) : t for col, t in types_convert.items() for i in iters }

        # Convert types
        df = df.astype(types_convert)

        # Print information in each column 
        cond_vars, sweep_vars = print_col_infos(df, additional_sweep_vars=args.additional_sweep_vars.split(","))
        groups = get_group_spec(args.groups, sweep_vars)

        has_modified_since = "modified_since" in df 

        if has_modified_since:
            print("Recent modified since: ", df["modified_since"].dropna().min())

        for col in key_stats:
            sel = [col, "folder", "_config_str", f"{col}_best_idx", f"{col}_len"]
            if has_modified_since:
                sel.append("modified_since")
            data = df[sel]
            if args.print_top_n:
                print_top_n(col, data, get_metric_info)

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

        if args.rows is not None and args.cols is not None:
            # Make Latex table. 
            row_names, row_choices = get_var_range(args.rows, sweep_vars)
            col_names, col_choices = get_var_range(args.cols, sweep_vars)

            for key_stat in key_stats:
                print(f"Table for {key_stat}. Rows: {args.rows}, Cols: {args.cols}")
                print(print_latex(df, row_names, row_choices, col_names, col_choices, key_stat, precision=args.precision))

            if args.groups == "/" or args.groups is None:
                groups = row_names + col_names 

        elif args.groups == "/":
            groups = reorder_group(groups)

        # Print out group means.
        if groups is not None:
            cols = [ col for col in key_stats ] + [ col + "_len" for col in key_stats ] + [ "_config_str", "folder" ]

            # Add aggregation function for each key_stats
            agg_obj = MeanStdAggFunc(precision=args.precision, use_latex=args.use_latex_agg)

            aggs = { col: [ agg_obj.agg ] for col in key_stats }
            aggs.update({ col + "_len": [ agg_obj.agg ] for col in key_stats })

            # Add folder aggregation. List all subfolder names for each breakdown category. 
            folder_agg_obj = FolderAggFunc()
            aggs["folder"] = [ folder_agg_obj.agg ]

            # convert _config_str to multiple columns, each is a hyper-parameter. 
            df = df[cols].apply(configstr2cols, axis="columns", args=(groups,))
            # Group according to the specified groups, and aggregated with aggration function. 
            df_agg = df.groupby(groups).agg(aggs)
            print(df_agg)

            if not args.use_latex_agg:
                c = df.groupby(groups).agg({ col : 'mean' })[col]
                print(f"max_val: {c.max()} at " + ",".join([f"{g}={v}" for g, v in zip(groups, c.idxmax())]))
                print(f"min_val: {c.min()} at " + ",".join([f"{g}={v}" for g, v in zip(groups, c.idxmin())]))

            if args.save_grouped_table:
                tbl_save = os.path.join(prefix, output_tbl_filename) 
                pickle.dump(df_agg, open(tbl_save, "wb"))
                print(f"*** output table saved in {tbl_save}") 

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
