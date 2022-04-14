import os
import sys
import numpy as np
import math
import utils
import json
import re
import pandas as pd

from utils import signature

def set_panda():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

def config2dict(s):
    return { item.split("=")[0]: item.split("=")[1] for item in s.split(",") } 

def config_filter(row, config_strs, iter_thres):
    for col, min_iter in iter_thres.items():
        if col in row:
            data = row[col]
            if isinstance(data, (float,int)):
                data = [data]
            if len(data) < min_iter:
               # This row is not processed. 
               return False

    if config_strs is None:
        return True

    config = json.loads(row["_config_str"])

    for k, v in config_strs.items():
        if config.get(k, None) != v:
            return False
    return True

def configstr2cols(row, groups):
    config = config2dict(row["_config_str"])
    for k in groups:
        row[k] = config.get(k, None)
    del row["_config_str"]
    return row

def combine_key_iter(col, i):
    return f"{col}_iter{i}"

def process_func(row, cols, params_per_col, first_k_iter):
    if cols is None:
        return row

    # For config str, remove 'githash' and 'sweep_filename'
    prefix = "override_"
    configs = []
    for key, value in row.items():
        if key.startswith(prefix) and not key[len(prefix):] in ('githash', 'sweep_filename'):
            configs.append(key[len(prefix):] + "=" + str(value).replace(",", "_"))

    row["_config_str"] = ",".join(configs)

    if first_k_iter is not None:
        iters = first_k_iter.split(",")
    else:
        iters = None

    for col in cols:
        descending = params_per_col(col, "descending")
        subkey = params_per_col(col, "subkey")
        topk_mean = params_per_col(col, "topk_mean")

        # Deal with multiple iterations.  
        if iters is not None:
            #if not isinstance(row[col], list):
            #    import pdb
            #    pdb.set_trace()
            all_data = [ row[col][:int(first_k_iter)] for first_k_iter in iters ]
            all_col_names = [ combine_key_iter(col, i) for i in iters ]
        else:
            all_data = [ row[col] ]
            all_col_names = [ col ]

        for data, col_name in zip(all_data, all_col_names):
            if isinstance(data, (int,float)):
                data = [data]

            if len(data) > 0:
                if isinstance(data[0], dict):
                    assert subkey is not None, "With list of dict as data, a subkey is needed!"
                    # Use subkey
                    data = [ d[subkey] for d in data ]
                data = np.array(data) 
                inds = data.argsort()
                if descending:
                    inds = inds[::-1]
                data = data[inds]

                best = data[0]
                best_idx = inds[0]

                if len(data) < topk_mean:
                    topk_mean_value = sum(data) / len(data)
                else:
                    topk_mean_value = sum(data[:topk_mean]) / topk_mean
            else:
                best = None
                best_idx = None
                topk_mean_value = None

            row[col_name] = topk_mean_value
            row[f"{col_name}_best"] = best
            row[f"{col_name}_best_idx"] = best_idx
            row[f"{col_name}_len"] = len(data)
    return row 
