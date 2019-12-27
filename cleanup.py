import os
import fnmatch
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolders", type=str, help="subfolders to check, separated by comma")
    parser.add_argument("--patterns", type=str)
    parser.add_argument("--switches", type=str)

    args = parser.parse_args()

    checkpoint_path = f"/checkpoint/{os.environ['USER']}/outputs" 

    patterns = args.patterns.split(",")
    switches = args.switches.split(",")

    print(patterns)
    print(switches)

    for pattern, switch in zip(patterns, switches):
        for subfolder in args.subfolders.split(","):
            print(f"Check {subfolder}")
            for root, dirnames, filenames in os.walk(os.path.join(checkpoint_path, subfolder)):
                filenames = [ os.path.join(root, filename) for filename in fnmatch.filter(filenames, pattern) ]
                if len(filenames) == 0: 
                    continue

                mtimes = map(lambda x : os.path.getmtime(x), filenames) 
                filenames = list(zip(mtimes, filenames))

                # sorted from newest to oldest.
                filenames = sorted(filenames, key=lambda x: -x[0])

                print(f"{root}: Files to keep: ")
                print(filenames[:2])

                # only keep the last three and delete the rest.
                for _, f in filenames[3:]:
                    os.remove(f)
        

if __name__ == "__main__":
    main()
