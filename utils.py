import os
import sys

def get_checkpoint_path():
    return f"/checkpoint/{os.environ['USER']}" 

def get_checkpoint_output_path():
    return f"/checkpoint/{os.environ['USER']}/outputs" 

def get_checkpoint_summary_path():
    return f"/checkpoint/{os.environ['USER']}/summary" 

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


