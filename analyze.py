from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from datetime import timedelta
import re
import time
import os
import sys
import torch
import json
import glob
import pandas as pd
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

class LogProcessor:
    def __init__(self):
        pass
 
    def _load_customized_processing(self, subfolder, args):
        if args.module_checkresult is None:
            main_file = MultiRunUtil.get_main_file(subfolder)
            main_file_checkresult = main_file + "_checkresult.py"
            if not os.path.exists(main_file_checkresult):
                main_file_checkresult = main_file + ".py"
        else:
            main_file_checkresult = args.module_checkresult

        spec = importlib.util.spec_from_file_location("", main_file_checkresult)
        mdl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mdl)

        default_attr_multirun = {
            "check_result": MultiRunUtil.load_inline_json            
        }

        attr_multirun = getattr(mdl, "_attr_multirun", default_attr_multirun)
        # Check attr_multirun
        return attr_multirun["check_result"](subfolder)

        # we import all names that don't begin with _ and main
        # names = [x for x in mdl.__dict__ if not x.startswith("_") and not x == "main"]

        # now drag them in
        # globals().update({k: getattr(mdl, k) for k in names})
        # return mdl._check_result(subfolder, args)

    def load_one(self, params):
        subfolder = params["subfolder"]
        args = params["args"]

        overrides = MultiRunUtil.load_cfg(subfolder)
        if len(overrides) == 0:
            return None

        config_str = ",".join(overrides)
        config = dict([ ("override_" + entry).split('=') for entry in overrides ])
        config["_config_str"] = config_str
        # print(config)

        first_group = None
        if params["first"] and "override_sweep_filename" in config:
            first_group = dict()
            sweep_filename = config.get("override_sweep_filename", '')
            if sweep_filename != '':
                for line in open(sweep_filename, "r"):
                    first_group["command"] = line.strip()
                    break

        entries = self._load_customized_processing(subfolder, args)

        if entries is None:
            return None

        for entry in entries:
            entry.update(config)

        if first_group is not None:
            entries[0]["_first"] = first_group

        return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdirs", type=str)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=utils.get_checkpoint_summary_path())
    parser.add_argument("--update_all", default=False, action="store_true", help="Update all existing summaries")
    parser.add_argument("--no_sub_folder", action="store_true")
    parser.add_argument("--wildcard_as_subfolder", type=str, default=None, help="can be '*.txt' etc")
    parser.add_argument("--module_checkresult", type=str, default=None, help="Specify the module .py file that contains _attr_multirun used to summarize the results.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    log_processor = LogProcessor()

    logdirs = []
    if args.update_all:
        for summary_file in glob.glob(os.path.join(args.output_dir, "*")):
            print(f"Collecting {summary_file} for updating")
            basename = os.path.basename(summary_file)
            basename = os.path.splitext(basename)[0]
            logdirs.append(basename.replace("_", "/"))
    
    logdirs = utils.parse_logdirs(args.logdirs)

    s = ""
    for curr_path in logdirs:
        print(f"Processing {curr_path}")
        curr_path = utils.preprocess_logdir(curr_path)
        df_name = curr_path.replace("/", "_")

        if args.no_sub_folder:
            subfolders = [ curr_path ]
        elif args.wildcard_as_subfolder is not None:
            subfolders = [ f for f in glob.glob(os.path.join(curr_path, args.wildcard_as_subfolder)) ] 
        else:
            subfolders = [ subfolder for subfolder in glob.glob(os.path.join(curr_path, "*")) if not subfolder.startswith('.') and os.path.isdir(subfolder) ]

        # test = log_processor.load_one(dict(subfolder=subfolders[0], args=args, first=True))
        res = []
        if args.num_process == 1:
            # Do not use multi-processing.
            for i, subfolder in tqdm.tqdm(enumerate(subfolders), total=len(subfolders)):
                entry = log_processor.load_one(dict(subfolder=subfolder, args=args, first= (i == 0)))
                if entry is not None:
                    res += entry
        else:
            pool = mp.Pool(args.num_process)
            try:
                num_folders = len(subfolders)
                chunksize = (num_folders + args.num_process - 1) // args.num_process
                print(f"Chunksize: {chunksize}")
                arguments = [ dict(subfolder=subfolder, args=args, log_converter=log_converter, first= (i == 0)) for i, subfolder in enumerate(subfolders) ]
                results = pool.imap_unordered(log_processor.load_one, arguments, chunksize=chunksize)
                for entry in tqdm.tqdm(results, total=num_folders):
                    if entry is not None:
                        res += entry

            except Exception as e:
                print(e)
                print(get_mem_usage())

        # find all folders starts with . (but not . and ..)
        meta = {
            "hidden": [ os.path.basename(f) for f in glob.glob(os.path.join(curr_path, ".??*")) ]
        }

        # Take the first information out.
        if "_first" in res[0]:
            meta.update(res[0]["_first"])
            del res[0]["_first"]

        df = pd.DataFrame(res)

        filename = os.path.join(args.output_dir, df_name + ".pkl")
        pickle.dump(dict(df=df, meta=meta), open(filename, "wb"))

        print(f"Size: {os.path.getsize(filename) / 2 ** 20} MB")
        print(f"Columns: {df.columns}")

        s += f"# {meta}\n"
        s += f"watch.append(\"{filename}\")\n"

    print(s)

if __name__ == "__main__":
    main()
