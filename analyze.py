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

filename_cfg_matcher = re.compile("[^=]+=[^=_]+")

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

    def _load_tensorboard(self, subfolder, args):
        event_files = [ glob.glob(os.path.join(subfolder, args.tb_folder, "stat.tb/*")) ]
        if args.tb_choice == "largest":
            # Use the largest event_file.
            files = [ (os.path.getsize(event_file), event_file) for event_file in event_files ] 
            files = sorted(files, key=lambda x: -x[0])[:1]
        elif args.tb_choice == "earliest":
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: x[0])[:1]
        elif args.tb_choice == "latest":
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: -x[0])[:1]
        elif args.tb_choice == "all":
            # All files, earliest first
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: x[0])
        else:
            raise RuntimeError(f"Unknown tb_choice: {args.tb_choice}")

        if len(files) == 0:
            return None

        entry = dict(folder=subfolder)
        for _, event_file in files:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            for key_name in ea.Tags()["scalars"]:
                l = [ s.value for s in ea.Scalars(key_name) ]
                if key_name not in entry: 
                    entry[key_name] = []
                entry[key_name] += l

        # Format: 
        # List[Dict[str, List[value]]]: number of trials * (key, a list of values)
        return [entry]

    def _load_checkpoint(self, subfolder, args):
        # [TODO] Hardcoded path.
        # sys.path.append("/private/home/yuandong/forked/luckmatters/catalyst")
        summary_file = os.path.join(subfolder, args.summary_file)

        stats = None
        for i in range(10):
            try:
                if os.path.exists(summary_file):
                    stats = torch.load(summary_file)
                    if hasattr(stats, "stats"):
                        stats = stats.stats
                break
            except Exception as e:
                time.sleep(2)

        if stats is None:
            # print(subfolder)
            # print(e)
            return None

        if not isinstance(stats, dict):
            stats = [stats]
        else:
            stats = stats.values()

        entries = []
        for one_stat in stats:
            if one_stat is not None:
                entry = dict(folder=subfolder)
                entry.update(listDict2DictList(one_stat))
                entries.append(to_cpu(entry))

        return entries

    def _load_log(self, subfolder, args):
        if os.path.isdir(subfolder):
            all_log_files = list(glob.glob(os.path.join(subfolder, "*.log")))
            if len(all_log_files) == 0:
                return None
            # First log file.
            log_file = all_log_files[0]
        else:
            log_file = subfolder

        sec_diff = time.time() - os.path.getmtime(log_file)
        td = timedelta(seconds=sec_diff)

        entry = defaultdict(list)
        entry["folder"] = subfolder
        entry["modified_since"] = str(td)

        has_one = False
        with open(log_file, "r") as f:
            for line in f:
                for d in self.log_converter:
                    m = d["match"].search(line)
                    if not m:
                        continue

                    for key_act, val_act in d["action"]:
                        entry[eval(key_act)].append(eval(val_act))
                    has_one = True

        if has_one:
            return [ dict(entry) ]
        else:
            return None

    def _load_json(self, subfolder, args):
        all_log_files = list(glob.glob(os.path.join(subfolder, "*.log")))
        if len(all_log_files) == 0:
            return None

        log_file = all_log_files[0]
        entry = defaultdict(list)
        entry["folder"] = subfolder

        cnt = 0
        with open(log_file, "r") as f:
            for line in f:
                index = line.find(args.json_prefix)

                if index < 0:
                    continue

                this_entry = json.loads(line[index + len(args.json_prefix):])

                for k, v in this_entry.items():
                    entry[k].append(v)

                cnt += 1

        if cnt > 0:
            return [ dict(entry) ]
        else:
            return None

    def _load_cfg(self, subfolder):
        ''' Return list [ "key=value" ] '''
        if not os.path.isdir(subfolder):
            # If it is not a dir, then just read from its filename.
            s = os.path.splitext(os.path.basename(subfolder))[0]
            cfg = [ m for m in filename_cfg_matcher.findall(s) ]
            cfg = [ v.strip("_") for v in cfg ]
            return cfg

        if os.path.exists(os.path.join(subfolder, ".hydra")):
            return yaml.safe_load(open(os.path.join(subfolder, ".hydra/overrides.yaml"), "r"))

        if os.path.exists(os.path.join(subfolder, "multirun.yaml")):
            multirun_info = yaml.safe_load(open(os.path.join(subfolder, "multirun.yaml")))
            return multirun_info["hydra"]["overrides"]["task"]

        # Last resort.. read *.log file directly to get the parameters.
        num_files = list(glob.glob(os.path.join(subfolder, "*.log")))
        assert len(num_files) > 0
        log_file = num_files[0]
        text = None
        for line in open(log_file):
            if text is None:
                if not line.startswith("[") and not line.startswith(" "):
                    text = line
            else:
                if line.startswith("["):
                    break
                text += line

        assert text is not None
        overrides = yaml.safe_load(text)
        # From dict to list of key=value
        return [ f"{k}={v}" for k, v in overrides.items() if not isinstance(v, (dict, list)) ]

    def load_one(self, params):
        subfolder = params["subfolder"]
        args = params["args"]
        self.log_converter = params["log_converter"]

        overrides = self._load_cfg(subfolder)
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

        if args.loader is None:
            # Try them one by one. 
            entries = self._load_tensorboard(subfolder, args)
            if entries is None:
                entries = self._load_json(subfolder, args)
            if entries is None:
                entries = self._load_checkpoint(subfolder, args)
            if entries is None:
                entries = self._load_log(subfolder, args)
        else:
            entries = eval(f"self._load_{args.loader}(subfolder, args)")

        if entries is None:
            return None

        for entry in entries:
            entry.update(config)

        if first_group is not None:
            entries[0]["_first"] = first_group

        return entries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--num_process", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default=utils.get_checkpoint_summary_path())
    parser.add_argument("--update_all", default=False, action="store_true", help="Update all existing summaries")
    parser.add_argument("--no_sub_folder", action="store_true")
    parser.add_argument("--path_outside_checkpoint", action="store_true")
    parser.add_argument("--loader", default=None, choices=["tensorboard", "json", "log", "checkpoint"])
    parser.add_argument("--json_prefix", default="json_stats: ")
    parser.add_argument("--tb_choice", default="largest", choices=["largest", "latest", "earliest", "all"])
    parser.add_argument("--tb_folder", type=str, default="stats")
    parser.add_argument("--log_regexpr_json", type=str, default=None)
    parser.add_argument("--wildcard_as_subfolder", type=str, default=None, help="can be '*.txt' etc")
    parser.add_argument("--summary_file", default="summary.pth", choices=["stats.pickle", "summary.pth", "checkpoint.pth.tar"])

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
    elif args.path_outside_checkpoint:
        logdirs = args.logdirs.split(",")
    else:
        logdirs = utils.parse_logdirs(args.logdirs)

    if args.log_regexpr_json is not None:
        data = json.load(open(args.log_regexpr_json, "r"))
        log_converter = [ dict(match=re.compile(d["match"]), action=d["action"]) for d in data ]
    else:
        log_converter = None

    s = ""
    for root in logdirs:
        print(f"Processing {root}")
        df_name = root.replace("/", "_")

        if not args.path_outside_checkpoint:
            curr_path = os.path.join(utils.get_checkpoint_output_path(), root)
        else:
            curr_path = root

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
                entry = log_processor.load_one(dict(subfolder=subfolder, args=args, log_converter=log_converter, first= (i == 0)))
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
