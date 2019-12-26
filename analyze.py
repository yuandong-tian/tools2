from tensorboard.backend.event_processing import event_accumulator
import re
import os
import sys
import glob
import pandas as pd
import yaml
import multiprocessing as mp 
import tqdm
import pickle
import argparse

class LogProcessor:
    def __init__(self):
        pass

    def load_one(self, subfolder):
        config = yaml.safe_load(open(os.path.join(subfolder, ".hydra/overrides.yaml"), "r"))
        config = dict([ ("override_" + entry).split('=') for entry in config ])

        stats = dict()
        # print(config)
        for event_file in glob.glob(os.path.join(subfolder, "stat.tb/*")):
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            entry = dict(folder=subfolder)
            
            for key_name in ea.Tags()["scalars"]:
                entry[key_name] = [ s.value for s in ea.Scalars(key_name) ]

            entry.update(config)

        return entry, stats

def main():
    checkpoint_path = f"/checkpoint/{os.environ['USER']}" 

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdirs", type=str)
    parser.add_argument("--output_dir", type=str, default=os.path.join(checkpoint_path, "summary"))
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
        logdirs = args.logdirs.split(",")


    for root in logdirs:
        print(f"Processing {root}")
        df_name = root.replace("/", "_")

        curr_path = os.path.join(checkpoint_path, "outputs", root) 

        # find all folders starts with . (but not . and ..)
        meta = {
            "hidden": list(glob.glob(os.path.join(curr_path, ".??*")))
        }

        subfolders = list(glob.glob(os.path.join(curr_path, "*")))
        # load_one(subfolders[0])

        pool = mp.Pool(32)
        res = []
        stats = []
        for entry, stat in tqdm.tqdm(pool.imap_unordered(log_processor.load_one, subfolders), total=len(subfolders)):
            res.append(entry)
            stats.append(stat)

        df = pd.DataFrame(res)

        filename = os.path.join(args.output_dir, df_name + ".pkl")
        pickle.dump(dict(df=df, meta=meta), open(filename, "wb"))
        print(f"Save to {filename}")


if __name__ == "__main__":
    main()
