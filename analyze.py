from tokenize import group
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from datetime import timedelta
import re
import time
import os
import sys
import glob
import pandas as pd
import torch
import yaml
import multiprocessing as mp
import tqdm
import pickle
import argparse
import utils
import importlib.util

from common_utils import MultiRunUtil

def to_cpu(x):
    if isinstance(x, dict):
        return { k : to_cpu(v) for k, v in x.items() }
    elif isinstance(x, list):
        return [ to_cpu(v) for v in x ]
    elif isinstance(x, torch.Tensor):
        return x.cpu()
    else:
        return x

def mem2str(num_bytes):
    assert num_bytes >= 0
    if num_bytes >= 2 ** 30:  # GB
        val = float(num_bytes) / (2 ** 30)
        result = "%.3f GB" % val
    elif num_bytes >= 2 ** 20:  # MB
        val = float(num_bytes) / (2 ** 20)
        result = "%.3f MB" % val
    elif num_bytes >= 2 ** 10:  # KB
        val = float(num_bytes) / (2 ** 10)
        result = "%.3f KB" % val
    else:
        result = "%d bytes" % num_bytes
    return result

def get_mem_usage():
    import psutil

    mem = psutil.virtual_memory()
    result = ""
    result += "available: %s\t" % (mem2str(mem.available))
    result += "used: %s\t" % (mem2str(mem.used))
    result += "free: %s\t" % (mem2str(mem.free))
    # result += "active: %s\t" % (mem2str(mem.active))
    # result += "inactive: %s\t" % (mem2str(mem.inactive))
    # result += "buffers: %s\t" % (mem2str(mem.buffers))
    # result += "cached: %s\t" % (mem2str(mem.cached))
    # result += "shared: %s\t" % (mem2str(mem.shared))
    # result += "slab: %s\t" % (mem2str(mem.slab))
    return result


# Turn list of dict to dict of list.
def listDict2DictList(stats):
    entry = dict()
    for i, stat in enumerate(stats):
        for k, v in stat.items():
            if k in entry:
                entry[k].append(v)
            else:
                # Alignment.
                entry[k] = [None] * i + [v]

        for k, v in entry.items():
            if isinstance(v, list):
                if len(v) < i + 1:
                    v.append(None)
    return entry

def get_group_choice(subfolder, args):
    # Pick which result_group we want to use to grep results.
    attr = load_attr_multirun(subfolder, args)
    sel_groups = args.sel_result_group.split(",") if args.sel_result_group is not None else attr["default_result_group"] 

    # only allow either a dp group or a event group
    sel_groups_df = [ attr["result_group"][key][1] for key in sel_groups if attr["result_group"][key][0] == "df" ]
    sel_groups_event = [ attr["result_group"][key][1] for key in sel_groups if attr["result_group"][key][0] == "event" ]
    if len(sel_groups_df) > 0:
        group_choice = ("df", sel_groups_df)
    else:
        group_choice = ("event", sel_groups_event)

    return group_choice

# Load customized processing. 
def load_attr_multirun(subfolder, args):
    mdl = MultiRunUtil.load_check_module(subfolder, filename=args.module_checkresult)
    attr_multirun = getattr(mdl, "_attr_multirun", None)
    assert attr_multirun is not None, f"_attr_multirun needs to be present for sweeping!" 
    return attr_multirun 

    # we import all names that don't begin with _ and main
    # names = [x for x in mdl.__dict__ if not x.startswith("_") and not x == "main"]

    # now drag them in
    # globals().update({k: getattr(mdl, k) for k in names})
    # return mdl._check_result(subfolder, args)

class LogProcessor:
    def __init__(self):
        pass
 
    def load_one(self, arguments):
        subfolder, args, group_choice = arguments 

        # Load overridden parameters. 
        overrides = MultiRunUtil.load_cfg(subfolder)

        group_choice = group_choice or get_group_choice(subfolder, args)

        config_str = ",".join(overrides)
        config = dict([ ("override_" + entry).split('=') for entry in overrides ])
        config["_config_str"] = config_str
        config["folder"] = subfolder
        # print(config)

        # Get log file. 
        log_file = MultiRunUtil.get_log_file(subfolder)
        config["modified_since"] = MultiRunUtil.get_modified_since(log_file)

        # Get longest sections. 
        lines = MultiRunUtil.get_logfile_longest_section(log_file)

        entries = []
        if group_choice[0] == "df":
            for matcher in group_choice[1]:
                entries.extend(MultiRunUtil.load_df(subfolder, lines, matcher))
        elif group_choice[0] == "event":
            entry = dict()
            for matcher in group_choice[1]:
                try:
                    new_entry = MultiRunUtil.load_regex(subfolder, lines, matcher)
                    if new_entry is not None:
                        entry.update(new_entry)
                except:
                    pass

            if len(entry) > 0:
                entries = [entry]
        else:
            raise NotImplementedError(f"{group_choice[0]} not implemented!")

        # Also add the sweep parameters, if these parameters are not set yet. 
        for entry in entries:
            for k, v in config.items():
                if k not in entry: 
                    entry[k] = v

        return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", type=str)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=utils.get_checkpoint_summary_path())
    parser.add_argument("--module_checkresult", type=str, default=None, help="Specify the module .py file that contains _attr_multirun used to summarize the results.")
    parser.add_argument("--sel_result_group", type=str, default=None, help="group of results to check, separated by comma, otherwise default in _attr_multirun")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    log_processor = LogProcessor()
    logdirs = utils.parse_logdirs(args.logdirs)

    s = ""
    for curr_path in logdirs:
        print(f"Processing {curr_path}")
        curr_path = utils.preprocess_logdir(curr_path)
        df_name = curr_path.replace("/", "_")

        if curr_path.find("*") >= 0 or curr_path.find("?") >= 0:
            subfolders = [ f for f in glob.glob(curr_path) if os.path.isdir(f) ]
        elif not os.path.exists(os.path.join(curr_path, "multirun.yaml")):
            # no subfolders
            subfolders = [ curr_path ]
        else:
            # regular subfolders. 
            subfolders = [ subfolder for subfolder in glob.glob(os.path.join(curr_path, "*")) if not subfolder.startswith('.') and os.path.isdir(subfolder) ]

        # call to get check_module for subfolder
        if len(subfolders) == 0:
            continue

        # Find out which result groups to use. 
        # Examplar result group in  _attr_multirun:  
        #  "result_group": {
        #     "stats_details" : ("df", _matcher),
        #     "stats_train_eval": ("event", _matcher_event),
        #     "stats_representation": ("event", _matcher_event2)
        # },
        # args.sel_result_group is used to filtered out any keys that are not useful 
        # note that if args.sel_result_group picks any df entry, then all the other entries will be neglected. 

        # test = log_processor.load_one(dict(subfolder=subfolders[0], args=args, first=True))
        res = []
        if args.num_process == 1:
            # Do not use multi-processing.
            group_choice = get_group_choice(subfolders[0], args)
            for subfolder in tqdm.tqdm(subfolders, total=len(subfolders)):
                entry = log_processor.load_one((subfolder, args, group_choice))
                res.extend(entry)
        else:
            pool = mp.Pool(args.num_process)
            try:
                num_folders = len(subfolders)
                chunksize = (num_folders + args.num_process - 1) // args.num_process
                print(f"Chunksize: {chunksize}")
                arguments = [ (subfolder, args, None) for subfolder in subfolders ] 
                results = pool.imap_unordered(log_processor.load_one, arguments, chunksize=chunksize)
                for entry in tqdm.tqdm(results, total=num_folders):
                    res.extend(entry)

            except Exception as e:
                print(e)
                print(get_mem_usage())

        df = pd.DataFrame(res)

        command = None
        if "override_sweep_filename" in df.columns and df.shape[0] > 0:
            sweep_filename = df["override_sweep_filename"][0]
            if sweep_filename != '':
                for line in open(sweep_filename, "r"):
                    command = line.strip()
                    break
        meta = {
            # find all folders starts with . (but not . and ..)
            "hidden": [ os.path.basename(f) for f in glob.glob(os.path.join(curr_path, ".??*")) ],
            "subfolders": subfolders,
            "command": command
        }

        filename = os.path.join(args.output_dir, df_name + ".pkl")
        pickle.dump(dict(df=df, meta=meta), open(filename, "wb"))

        print(f"Size: {os.path.getsize(filename) / 2 ** 20} MB")
        print(f"Columns: {df.columns}")

        # s += f"# {meta}\n"
        s += f"watch.append(\"{filename}\")\n"

    print(s)

if __name__ == "__main__":
    main()
