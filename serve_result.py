import argparse
from collections import OrderedDict
from subprocess import check_output 
import time
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('list_file', type=str)
parser.add_argument('--check_freq', type=int, default=60, help="check frequency (in sec)")
parser.add_argument('--match_file', type=str, default=None)
parser.add_argument('--key_stats', type=str, default="acc")
parser.add_argument('--descending', action="store_true")
parser.add_argument('--output_prefix', type=str, default="output")

args = parser.parse_args()

abs_path = os.path.dirname(os.path.abspath(__file__))

while True:
    print(f"{datetime.now()}: Loading from {args.list_file}")
    state = 0
    records = OrderedDict()
    for line in open(args.list_file, "r"):
        line = line.strip(' ').strip("\n")
        if line == "":
            continue

        if state == 0:
            title = line
            state = 1
        elif state == 1:
            records[title] = line
            state = 0

    print(records)
        
    # Then draw the records to a file and serve it. 
    f = open(args.output_prefix + ".md", "w")
    f.write(f"# {datetime.now()} \n\n")
    for title, r in records.items():
        cmd = f"python {abs_path}/analyze.py --logdirs {r} --log_regexpr_json {args.match_file} --loader=log --num_process 1" 
        print(cmd)
        check_output(cmd, shell=True)
        cmd = f"python {abs_path}/stats.py --logdirs {r} --key_stats {args.key_stats} --topk_mean 1 --groups / "
        if args.descending:
            cmd += "--descending"
        print(cmd)
        output = check_output(cmd, shell=True).decode('utf-8')
        f.write(f"## {title}\n\n")
        f.write(f"```\n{output}\n```\n\n")
    f.close()

    print(f"{datetime.now()}: Sleep for {args.check_freq}s")
    time.sleep(args.check_freq)

        
    
