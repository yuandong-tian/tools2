import re
import json
from datetime import datetime
from subprocess import check_output
import os
import sys
import glob

from utils_sweeper import ParamHandler

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

# provide a script. JOB_IDX will be an environment variable to provide relative job idx

run_script = '''#!/bin/bash

# printenv | grep SLURM
echo slurm_node:$SLURMD_NODENAME slurm_job_id:$SLURM_JOB_ID cuda:$CUDA_VISIBLE_DEVICES job_idx:$JOB_IDX
'''

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("param_file", type=str)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--param_args", type=str, default="{}")
    parser.add_argument('--first_n_group', type=int, default=-1)

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--print_only", action="store_true")

    parser.add_argument('--num_jobs_per_group', type=int, default=8, help="if 0, then run all jobs in a single group")

    parser.add_argument('--time', type=str, default="12:00:00")
    parser.add_argument('--gres', type=str, default="gpu:1")
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument("--ntasks_per_node", type=int, default=1)
    parser.add_argument('--partition', type=str, default="learnfair")

    args = parser.parse_args()

    if args.name is None:
        args.name = os.path.dirname(args.param_file)

    output_matcher = re.compile(r"Submitted batch job ([0-9]+)")

    job_path = os.path.join(root, args.name)
    if os.path.exists(job_path):
        num_pickle = len(list(glob.glob(os.path.join(job_path, "*.pickle"))))
        num_out = len(list(glob.glob(os.path.join(job_path, "*.out"))))
        answer = input(f"{job_path} has {num_pickle} .pickle files and {num_out} .out files, do you want to delete?")
        if answer == "Y":
            import shutil
            shutil.rmtree(job_path)
        else:
            sys.exit(0)
    os.mkdir(job_path)

    param_args = eval(args.param_args)
    param_args["job_path"] = job_path
    handler = ParamHandler(args.param_file, args.num_jobs_per_group, param_args=param_args)

    job_desc = {
        "root": root,
        "name" : args.name,
        "dry_run": args.dry_run,
        "start_time": str(datetime.now()), 
        "jobs": []
    }

    num_group = handler.get_num_groups()
    if args.first_n_group > 0:
        print(f"Only run first {args.first_n_group} / {num_group}")
        num_group = min(num_group, args.first_n_group)

    print(f"#jobs: {handler.get_num_jobs()}, #group: {num_group}")
    for i in range(num_group):
        content = ""
        tmp_filename = "tmp-%d.sh" % i

        content += "#!/bin/bash\n"
        params = get_params(job_path, args, i)
        for k, v in params.items():
            content += "#SBATCH --%s=%s\n" % (k, v)
        content += module_load + "\n"
        for arg in handler.get_group(i):
            content += "srun %s\n" % handler.get_exec(arg)

        if args.print_only:
            print(content)
            continue

        tmp_filename = "tmp-%d.sh" % i
        with open(tmp_filename, "w") as f:
            f.write(content)

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
        
        
