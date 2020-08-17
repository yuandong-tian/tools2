import sys
import os
import itertools
import argparse
from subprocess import check_output

parser = argparse.ArgumentParser(description='')
parser.add_argument('--target_dir', default=None, type=str, help="Target dir")
parser.add_argument('--create_context', action="store_true", default=False)
parser.add_argument('command', type=str, help="Sweep command")

args = parser.parse_args()

options = []
tokens = args.command.split(" ")
for token in tokens:
    # For each token, convert anything like "=1,2" into "=1" and "=2".
    items = token.split("=")
    if len(items) == 1:
        options.append([ dict(argument=items[0],n=1) ])
        continue

    b, e = items

    # File expansion
    if e.startswith('{') and e.endswith('}'):
        with open(os.path.expanduser(e[1:-1]), "r") as f:
            e = f.readlines()
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
    except FileExistsError:
        pass

    with open(os.path.join(args.target_dir, "_context.log"), "w") as f:
        f.write(args.command + "\n")
        f.write(f"{n} jobs\n")
        f.write(check_output("git -C ./ log --pretty=format:'%H' -n 1", shell=True).decode('utf-8'))
        f.write(check_output("git diff", shell=True).decode('utf-8'))
        f.write(check_output("git diff --cached", shell=True).decode('utf-8'))
        

