import os
import sys
import torch

from datetime import datetime
import pickle
import glob
import re
import multiprocess
import argparse
import time
import random

import numpy as np

root = "/checkpoint/yuandong/jobs"

def signature():
    return str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")

def sig():
    return datetime.now().strftime("%m%d%y_%H%M%S_%f")

def print_info():
    if "SLURM_JOB_ID" in os.environ:
        print("slurm mode: " + os.environ.get("SLURMD_NODENAME", ""))
        print("slurm job id: " + os.environ.get("SLURM_JOB_ID", ""))
        print("cuda: " + os.environ.get("CUDA_VISIBLE_DEVICES", ""))

def add_parser_argument(parser):
    parser.add_argument("--save_dir", type=str, default="./")

def set_args(argv, args):
    cmdline = " ".join(argv)
    signature = sig()
    setattr(args, 'signature', signature)
    setattr(args, "cmdline", cmdline)

def save_data(prefix, args, data):
    filename = f"{prefix}-{args.signature}"
    save_dir = os.path.join(args.save_dir, filename) 
    print(f"Save to {save_dir}")

    pickle.dump(dict(data=data, args=args, save_dir=save_dir), open(save_dir + ".pickle", "wb"))
    pickle.dump(dict(args=args, save_dir=save_dir), open(save_dir + ".arg", "wb"))

def convert_cuda(a):
    if isinstance(a, list):
        return [ convert_cuda(aa) for aa in a ]
    elif isinstance(a, dict):
        return { k : convert_cuda(aa) for k, aa in a.items() }
    elif isinstance(a, torch.Tensor):
        return a.cpu()
    else:
        return a

def _load_jobs(fs):
    return [ convert_cuda(pickle.load(open(f, "rb"))) for f in fs ]

def _load_one_job(f):
    return convert_cuda(pickle.load(open(f, "rb")))
    # Check if anything is in cuda.

# Visualization side. 
def get_stats(jobnames, root=root, name_filter=None, num_process=32, ignore_summary=False):
    all_stats = []
    for job in jobnames:
        if not ignore_summary:
            summary_file = os.path.join(root, job, "summary.pickle") 
            if os.path.exists(summary_file):
                all_stats += pickle.load(open(summary_file, "rb"))
                continue

        filenames = []
        # file_buckets = [ list() for i in range(n) ]
        for filename in glob.glob(os.path.join(root, job, "*.pickle")):
            if name_filter is None or re.search(name_filter, filename) is not None:
                # idx = random.randint(0, n - 1)
                # file_buckets[idx].append(filename)
                filenames.append(filename)

        p = multiprocess.Pool(num_process)
        import tqdm

        all_stats += [ stats for stats in tqdm.tqdm(p.imap(_load_one_job, filenames), ncols=100) ]
                
        #for stats in tqdm.tqdm(p.imap(_load_one_job, filenames), ncols=100):
        #    all_stats.append(stats)
        # all_stats += [ stats for stats in p.imap(_load_one_job, filenames) ]

    return all_stats


class PlotData:
    def __init__(self, name):
        self.d = dict()
        self.name = name
        self.display_multi_curve = 'std'
    
    def add(self, d):
        # Add to key value pairs
        # all v needs to be numpy array. 
        for k, v in d.items():
            if k not in self.d:
                self.d[k] = [v]
            else:
                # save them
                self.d[k].append(v)
    
    def plot(self, plt, k1, k2s, colors, name2label=dict()):
        assert k1 in self.d
        
        x = self.d[k1][0]
        
        for k2, c in zip(k2s, colors):
            y = self.d[k2]
            label = name2label.get(k2, k2)
            if len(y) > 1:
                # Multiple series. Compute their min/max.
                ys = np.stack(y, axis=1)
                mean_y = np.mean(ys, axis=1)
                plt.plot(x, mean_y, color=c, label=label)
                
                if self.display_multi_curve == 'std':
                    std_y = np.std(ys, axis=1)
                    # draw error bar. 
                    plt.fill_between(x, mean_y - std_y, mean_y + std_y, color=c, alpha=0.2)
                elif self.display_multi_curve == 'minmax':
                    min_y = np.min(ys, axis=1)
                    max_y = np.max(ys, axis=1)
                    plt.fill_between(x, max_y, min_y, color=c, alpha=0.2)
            else:
                plt.plot(x, y[0], color=c, label=label)
            
    def __getitem__(self, k):
        return self.d[k]

    
class ArgFilter:
    def __init__(self, **kwargs):
        self.spec = kwargs
        
    def __getitem__(self, k):
        return self.spec[k]
        
    def __hash__(self):
        return hash(tuple(sorted(self.spec.items())))
        
    def info(self):
        return "-".join([ f"{k}={v}" for k, v in self.spec.items()])
    
    def check(self, args):
        if isinstance(args, ArgFilter):
            d = args.spec
        elif isinstance(args, dict):
            d = args
        else:
            d = args.__dict__
            
        for k, v in self.spec.items():
            if not k in d or d[k] != v:
                return False

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--num_process', type=int, default=32)
    parser.add_argument('--override_summary', action="store_true")

    args = parser.parse_args()
    t0 = time.time()

    summary_file = os.path.join(root, args.job_dir, "summary.pickle")
    if os.path.exists(summary_file):
        answer = input(f"{args.job_dir} has summary file, delete?")
        if answer != "Y":
            sys.exit(0)
        else:
            os.remove(summary_file)
        
    stats = get_stats([args.job_dir], num_process=args.num_process, ignore_summary=True)
    pickle.dump(stats, open(summary_file, "wb"))
    print(time.time() - t0)
