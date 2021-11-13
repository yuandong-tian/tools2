import os
import sys
from datetime import datetime
import yaml
import re

config = yaml.load(open(os.path.expanduser("~/.tools2.yaml")))
output_dir_matcher = re.compile(r"Sweep output dir\s*:\s*(.*)$")

def get_checkpoint_output_path():
    return config["output_path"]

def get_checkpoint_summary_path():
    return config["summary_path"]

def parse_logdirs(logdirs):
    checkpoint_output_path = get_checkpoint_output_path()

    if isinstance(logdirs, str):
        logdirs = logdirs.split(",")

    res = []
    for d in logdirs:
        if d.endswith(".log"):
            # A file. Open it and find Sweep dir
            for line in open(d):
                m = output_dir_matcher.search(line)
                if m:
                    d = m.group(1)
                    break

        # If d is absolute path, keep it. 
        # If d is relative, check whether it is present in the current folder, 
        #    if so, keep, otherwise connect with the checkpoint_output_path
        if d[0] != '/':
            for candidate_root in [checkpoint_output_path, os.getcwd()]:
                full_path = os.path.join(candidate_root, d)
                if os.path.exists(full_path):
                    res.append(full_path)
                    break
        else:
            res.append(d)

    return res

def preprocess_logdir(logdir):
    if not os.path.isdir(logdir):
        # Grab a line with "sweep output folder" and match it 
        subfolder_matcher = re.compile(r"sweep output dir : (.*)$")
        real_folder = None
        for line in open(logdir, "r"):
            m = subfolder_matcher.search(line)
            if m:
                real_folder = m.group(1)
                break
        assert real_folder is not None, f"{logdir} as a file, should contain real folder but it cannot be found!" 
        logdir = real_folder
        print(f"Redirect to {logdir}")
    return logdir


def signature():
    return str(datetime.now()).replace(" ", "_").replace(":", "-").replace(".", "_")

def compute_avg(epoch, values, span=20):
    minstep = min(max(epoch - span, 0), len(values))
    maxstep = min(epoch + span + 1, len(values))
    if minstep == maxstep:
        return None
        # raise RuntimeError(f"error! {i}")

    return sum(values[minstep:maxstep]) / (maxstep - minstep)


def shorten_folder(folder_name):
    return int(os.path.basename(folder_name))


def preprocess(data, eval_keys):
    df = data["df"]
    meta = data["meta"] 
    
    orig_factors = [ col for col in df.columns if col.startswith("override_") and len(df[col].unique()) > 1 ] 
    orig_nonfactors = [ col for col in df.columns if col.startswith("override_") and len(df[col].unique()) == 1 ] 
    
    factors = [ col[9:] for col in orig_factors ]
    nonfactors = [ col[9:] for col in orig_nonfactors ]
    
    df2 = df[orig_factors + eval_keys].rename(columns=dict(zip(orig_factors, factors)))
    
    nonfactors = { col[9:] : df[col].unique()[0] for col in orig_nonfactors }

    df2["folder"] = df["folder"].apply(shorten_folder)
    # replacing true with 1 and false with 0
    df2 = df2.replace({"true" : "1", "false" : "0"})
    return df2, nonfactors, factors


def display_table(df, factors, y_prefixes, all_num=True):  
    # data2.astype('float32').groupby(["max_importance_ratio", "grad_clip", "lr"]).mean()

    # data3 = data2[data2["lr"] == "0.0001"]
    # data3 = data2[data2["trainer.params.actor_sync_freq"] == "25"]
    # data3 = data2[data2["negative_momentum"] == "-0.2"]
    # data3 = data2[ (data2["lr"] == "0.0001") | (data2["lr"] == "0.0002") ]
    for f in factors:
        print(f"Factor: {f}")
        if all_num:
            tmp = df.astype('float32')
        else:
            tmp = df
        # tmp = df.astype('float32').groupby([f]).mean()
        tmp = tmp.groupby([f]).mean()
        
        for y_prefix in y_prefixes:
            tmp2 = tmp[[ col for col in tmp.columns if col.startswith(y_prefix)]]
            display(tmp2)
            tmp2.T.plot.line(figsize=(20,15), title=f"{f}-{y_prefix}")
