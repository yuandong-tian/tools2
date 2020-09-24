import sys
import re
import os
import itertools
import argparse
from subprocess import check_output
from time import sleep

'''
Tools to run C++ sweeps. 

One example:
    python ~/tools2/gen_sweep.py "./jps --game=simplehanabi --iter=100 --iter_cfr=100 --seed={seq 1 100} --use_cfr_pure_init" --target_dir simplehanabi --create_context | parallel -j60
'''

parser = argparse.ArgumentParser(description='')
parser.add_argument('--target_dir', default=None, type=str, help="Target dir")
parser.add_argument('--create_context', action="store_true", default=False)
parser.add_argument('command', type=str, help="Sweep command")

args = parser.parse_args()

options = []
tokens = re.findall(r"([^\s]+=\{.*\}|[^\s]+)", args.command)

for token in tokens:
    # For each token, convert anything like "=1,2" into "=1" and "=2".
    items = token.split("=", 2)
    if len(items) == 1:
        options.append([ dict(argument=items[0],n=1) ])
        continue

    b, e = items

    # Execute the command..
    if e.startswith('{') and e.endswith('}'):
        # execute the command. 
        res = check_output(e[1:-1], shell=True).decode('utf-8')
        e = re.split(r"[,\s\n]\s*", res.strip())
    else:
        e = e.split(",") 

    e = [ ee.strip() for ee in e ]

    options.append([ dict(n=len(e), abbr=f"{b.strip('-')}={ee if len(str(ee)) < 20 else i}", argument=f"{b}={ee}") for i, ee in enumerate(e) if ee != "" ])

# Enumerate all possible combinations and output.
n = 0
for i, entry in enumerate(itertools.product(*options)):
    command = " ".join([ e["argument"] for e in entry ])
    prefix = "_".join([ e["abbr"] for e in entry if e["n"] > 1])
    if args.target_dir is not None:
        command += f" > {args.target_dir}/{prefix}.txt" 
    print(command)
    n += 1

if args.create_context:
    # Create context.
    assert args.target_dir is not None
    try:
        os.mkdir(args.target_dir)
        sleep(5)
    except FileExistsError:
        pass

    with open(os.path.join(args.target_dir, "_context.log"), "w") as f:
        f.write(args.command + "\n")
        f.write(f"{n} jobs\n")
        f.write(check_output("git -C ./ log --pretty=format:'%H' -n 1", shell=True).decode('utf-8'))
        f.write(check_output("git diff", shell=True).decode('utf-8'))
        f.write(check_output("git diff --cached", shell=True).decode('utf-8'))
        

