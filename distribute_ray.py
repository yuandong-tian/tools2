import json
import subprocess
import shlex
import glob
import sys
import os
import re
import tqdm
import time
from datetime import datetime

def run(cmdline):
    subprocess.call(cmdline, shell=True)
    return cmdline
    
# each line is a task and we want to run it.
tasks = []
for line in open(sys.argv[1], "r"):
    if line.startswith("exec"):
        output = subprocess.check_output(shlex.split(line[4:])).decode()
        tasks += output.strip().split("\n")
    else:
        tasks.append(line)

curr_path = os.getcwd()
tasks = [ f"cd {curr_path}; {task}" for task in tasks ]

from ray.util.multiprocessing.pool import Pool # NOTE: Only the import statement is changed.
pool = Pool()

print(f"#Tasks: {len(tasks)}")
for task in tasks[:min(len(tasks), 5)]:
    print(task)

# Then run ray. 
start = time.time()
for result in tqdm.tqdm(pool.map(run, tasks)):
    pass

print("Finished in: {:.2f}s".format(time.time()-start))
