import os
import sys
import yaml

config = yaml.load(open(os.path.expanduser("~/.tools2.yaml")))

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
        if d.startswith(checkpoint_output_path):
            d = d[len(checkpoint_output_path) + 1:]
        res.append(d)

    return res

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
