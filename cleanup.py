import os
import fnmatch
import sys
import argparse
from pathlib import Path

def keep_recent(filenames, recent_n):
    mtimes = map(lambda x : os.path.getmtime(x), filenames) 
    filenames = list(zip(mtimes, filenames))

    # sorted from newest to oldest.
    filenames = sorted(filenames, key=lambda x: -x[0])

    if recent_n > 0:
        print("Files to keep: ")
        print(filenames[:recent_n])
    else:
        print("Delete all")

    # only keep the last three and delete the rest.
    for _, f in filenames[recent_n:]:
        os.remove(f)

def display(filenames):
    print(filenames[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolders", type=str, default="./", help="subfolders to check, separated by comma")
    parser.add_argument("--patterns", type=str)
    parser.add_argument("--op", type=str, default=None, help="")

    args = parser.parse_args()
    
    if args.op is None:
        print("--op needs to specify concrete actions (e.g., keep=3, display)")
        os.exit(0)

    checkpoint_path = f"/checkpoint/{os.environ['USER']}/outputs" 

    patterns = args.patterns.split(",")
    print(patterns)

    empty_folders = []

    for subfolder in args.subfolders.split(","):
        print(f"Check {subfolder}")
        for root, dirnames, filenames in os.walk(os.path.join(checkpoint_path, subfolder)):
            if len(dirnames) == 0 and len(filenames) == 0:
                empty_folders.append(root)

            for pattern in patterns:
                filenames = [ os.path.join(root, filename) for filename in fnmatch.filter(filenames, pattern) ]
                if len(filenames) == 0: 
                    continue

                for op in args.op.split(","):
                    items = op.split("=") 
                    eval(items[0] + "(filenames, " + ",".join(items[1:]) + ")")

    print("Empty folders:")
    print(empty_folders)
    

if __name__ == "__main__":
    main()
