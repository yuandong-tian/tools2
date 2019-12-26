import numpy as np
import os
import glob
import json

from .matchers import ParallelParser
from collections import defaultdict

def get_log_fullname(root, job_id, job_idx, inputs, check=False):
    f = "%d/output%d-0.log" % (job_id, job_idx)
    prefix = "%d-%d" % (job_id, job_idx)
    f = os.path.join(root, f)
    if not check or os.path.exists(f):
        inputs.append((prefix, f))
        return True

def get(job_names):
    root = "/checkpoint/yuandong/jobs"
    inputs = []
    for job_name in job_names:
        for f in glob.glob(os.path.join(root, job_name, "*.out")):
            prefix = job_name + "-" + os.path.basename(f)
            inputs.append((prefix, f))

    return parse_files(inputs)

def get_aml(job_names):
    json_root = "/home/yuandong/tools/sweeper/jobs"
    root = "/mnt/vol/gfsai-flash-east/ai-group/users/yuandong/rts"

    inputs = []
    for job_name in job_names:
        if not job_name.endswith(".json"):
            job_name += ".json"

        full_name = os.path.join(json_root, job_name)
        if not os.path.exists(full_name):
            print(f"{full_name} is not found!")
            continue

        with open(full_name) as f:
            jobs = json.load(f)

        if "jobs" in jobs:
            for job in jobs["jobs"]:
                if "id" in job:
                    get_log_fullname(root, job["id"], job["job_idx"], inputs)
        else:
            # Treat jobs as a list.
            for job_id in jobs:
                job_idx = 0
                while True:
                    if not get_log_fullname(root, job_id, job_idx, inputs, check=True):
                        break
                    job_idx += 1

    return inputs

import pandas as pd

def resample(df, at, method='linear'):
    # previously it was the next line. Change it to take the union of new and old
    # df = df.reindex(numpy.arange(df.index.min(), df.index.max(), 0.0005))
    df = df.reindex( (df.index | at).unique() )
    df = df.interpolate(method=method).loc[at]
    return df

def process(data, factors, xlabel, xs, ylabel, conditions={}):
    # Merge runs whose factors are the same.
    processed = dict()
    for k, v in data.items():
        args = v["args"][0]

        key = "-".join([ "%s=%s" % (f, str(args[f])) for f in factors])
        if any([args[f] != value for f, value in conditions.items()]):
            continue

        # print("Processing %s, key: %s" % (k, key))
        df = pd.DataFrame(v["stats"]).set_index(xlabel).rename(columns={ ylabel : ylabel + "_" + k })
        df = df[~df.index.duplicated(keep='first')]
        df = resample(df, xs)
        if key not in processed:
            processed[key] = df
        else:
            processed[key] = processed[key].join(df)

    result = dict()
    for key, df in processed.items():
        result[key] = pd.DataFrame(dict(mean=df.mean(axis=1), std=df.std(axis=1)))

    return result, processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_files', type=str, nargs='+', help="Json files")
    parser.add_argument("--matcher", type=str, default="standard_matcher", help="Match string")

    args = parser.parse_args()

    parser = ParallelParser(args.matcher)
    inputs = get_aml(args.json_files)
    data = parser.parse(inputs)
    print(data.keys())

