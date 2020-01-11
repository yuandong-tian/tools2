from tensorboard.backend.event_processing import event_accumulator
import re
import time
import os
import sys
import torch
import glob
import pandas as pd
import yaml
import multiprocessing as mp 
import tqdm
import pickle
import argparse
import utils

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

class LogProcessor:
    def __init__(self):
        pass

    def _load_tensorboard(self, subfolder):
        entry = None
        for event_file in glob.glob(os.path.join(subfolder, "stat.tb/*")):
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            entry = dict(folder=subfolder)
            
            for key_name in ea.Tags()["scalars"]:
                entry[key_name] = [ s.value for s in ea.Scalars(key_name) ]

        return entry

    def _load_checkpoint(self, subfolder):
        # [TODO] Hardcoded path. 
        sys.path.append("/private/home/yuandong/forked/luckmatters/catalyst")
        summary_file = os.path.join(subfolder, "summary.pth")
        checkpoint_file = os.path.join(subfolder, "checkpoint.pth.tar")

        stats = None
        for i in range(10):
            try:
                if os.path.exists(summary_file):
                    stats = torch.load(summary_file)
                elif os.path.exists(checkpoint_file):
                    checkpoint = torch.load(checkpoint_file)
                    stats = checkpoint.stats
                break
            except Exception as e:
                time.sleep(2)

        if stats is None:
            # print(subfolder)
            # print(e)
            return None

        # Turn list of dict to dict of list. 
        entry = dict(folder=subfolder)
        for i, stat in enumerate(stats):
            for k, v in stat.items():
                if k in entry:
                    entry[k].append(v)
                else:
                    # Alignment. 
                    entry[k] = [None] * i + [v]

            for k, v in entry.items():
                if len(v) < i + 1:
                    v.append(None)

        return to_cpu(entry)

    def load_one(self, subfolder):
        config = yaml.safe_load(open(os.path.join(subfolder, ".hydra/overrides.yaml"), "r"))
        config = dict([ ("override_" + entry).split('=') for entry in config ])
        # print(config)

        entry = self._load_tensorboard(subfolder)
        if entry is None:
            entry = self._load_checkpoint(subfolder)

        if entry is not None:
            entry.update(config)
        return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--num_process", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=utils.get_checkpoint_summary_path())
    parser.add_argument("--update_all", default=False, action="store_true", help="Update all existing summaries")

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
    else:
        logdirs = utils.parse_logdirs(args.logdirs)

    s = ""
    for root in logdirs:
        print(f"Processing {root}")
        df_name = root.replace("/", "_")

        curr_path = os.path.join(utils.get_checkpoint_output_path(), root) 

        # find all folders starts with . (but not . and ..)
        meta = {
            "hidden": [ os.path.basename(f) for f in glob.glob(os.path.join(curr_path, ".??*")) ]
        }

        subfolders = list(glob.glob(os.path.join(curr_path, "*")))
        # load_one(subfolders[0])

        res = []

        if args.num_process == 1:
            # Do not use multi-processing.
            for subfolder in tqdm.tqdm(subfolders, total=len(subfolders)):
                entry = log_processor.load_one(subfolder) 
                if entry is not None:
                    res.append(entry)
        else:
            pool = mp.Pool(args.num_process)
            try:
                num_folders = len(subfolders)
                chunksize = (num_folders + args.num_process - 1) // args.num_process
                print(f"Chunksize: {chunksize}")
                results = pool.imap_unordered(log_processor.load_one, subfolders, chunksize=chunksize)
                for entry in tqdm.tqdm(results, total=num_folders):
                    if entry is not None:
                        res.append(entry)

            except Exception as e:
                print(e)
                print(get_mem_usage())

        df = pd.DataFrame(res)

        filename = os.path.join(args.output_dir, df_name + ".pkl")
        pickle.dump(dict(df=df, meta=meta), open(filename, "wb"))

        print(f"Size: {os.path.getsize(filename) / 2 ** 20} MB")
        print(f"Columns: {df.columns}")

        s += f"# {meta['hidden']}\n" 
        s += f"watch.append(\"{filename}\")\n"

    print(s)

if __name__ == "__main__":
    main()
