import re
import json
from datetime import datetime
from subprocess import check_output
import os

# Examples:
#SBATCH --job-name=theory
#SBATCH --output=/checkpoint/%u/jobs/theory-%j.out
#SBATCH --error=/checkpoint/%u/jobs/theory-%j.err

#SBATCH --gres=gpu:1
#SBATCH --partition=uninterrupted
#SBATCH --nodes=1
#SBATCH --time=03:00:00

root = "/checkpoint/yuandong/jobs" 

def get_params(job_path, args, job_idx):
    params = {
        "job-name" : args.name,
        "output": os.path.join(job_path, "%d.out" % job_idx),
        "error": os.path.join(job_path, "%d.err" % job_idx),
    }

    keys = ["gres", "partition", "nodes", "time", "ntasks-per-node"]
    for k in keys:
        params[k] = getattr(args, k.replace("-", "_"))
    return params


module_load = '''
# Start clean
#module purge

#module load cuda/9.0
#module load cudnn/v7.0-cuda.9.0

export PATH=$HOME/miniconda3/bin:$PATH

module unload gcc/5.2.0

source ~/lmod_go.sh
source activate go10

#module load anaconda3/5.0.1
#module load torch/012218/gcc.5.4.0
'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument('--num_jobs', type=int, default=1)
    parser.add_argument('--command', type=str, default="run.sh", 
            help="provide a script. JOB_IDX will be an environment variable to provide relative job idx")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument('--time', type=str, default="3:00:00")
    parser.add_argument('--gres', type=str, default="gpu:1")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument("--ntasks_per_node", type=int, default=1)
    parser.add_argument('--partition', type=str, default="dev")

    args = parser.parse_args()

    output_matcher = re.compile(r"Submitted batch job ([0-9]+)")

    job_path = os.path.join(root, args.name)
    if os.path.exists(job_path):
        print("JobPath exists: %s" % job_path)
        os.exit(1)

    os.mkdir(job_path)

    job_desc = {
        "root": root,
        "name" : args.name,
        "dry_run": args.dry_run,
        "start_time": str(datetime.now()), 
        "jobs": []
    }

    for i in range(args.num_jobs):
        tmp_filename = "tmp-%d.sh" % i
        with open(tmp_filename, "w") as f:
            f.write("#!/bin/bash\n")
            params = get_params(job_path, args, i)
            for k, v in params.items():
                f.write("#SBATCH --%s=%s\n" % (k, v))
            f.write(module_load)
            f.write("JOB_IDX=%d srun --label %s\n" % (i, args.command))

        if not args.dry_run:
            output = check_output(["sbatch", tmp_filename]).decode('utf-8').strip()
            print(output)
            m = output_matcher.match(output)
            os.remove(tmp_filename)
        else:
            m = None

        if m is None: 
            job_desc["jobs"].append(None)
        else:
            job_desc["jobs"].append(int(m.group(1)))

    with open(os.path.join(job_path, "summary.json"), "w") as f:
        json.dump(job_desc, f)
        
        
