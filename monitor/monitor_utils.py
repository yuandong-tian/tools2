from subprocess import Popen, check_output
import json
import os
import re

comment_matcher = re.compile("--comment=([^ ]+)")
stderr_matcher = re.compile("-e ([^ ]+)")
stdout_matcher = re.compile("-o ([^ ]+)")

def get_git_hash():
    try:
        return check_output("git -C ./ log --pretty=format:'%H' -n 1", shell=True).decode('utf-8')
    except:
        return ""

def vis_seq(ss):
    if len(ss) == 0:
        return ""

    ss = sorted(ss)
    # [6,7,8,10,12,13] -> 6-8,10,12-13
    last_start = ss[0]
    last_end = ss[0]
    output = []
    for s in ss[1:]:
        if s > last_end + 1:
            if last_end > last_start:
                output.append(f"{last_start}-{last_end}")
            else:
                output.append(f"{last_start}")
            last_start = last_end = s
        else:
            last_end = s

    if last_end > last_start:
        output.append(f"{last_start}-{last_end}")
    else:
        output.append(f"{last_start}")

    return ",".join(output) + f"[{len(ss)}]"

def load_seq(s):
    # load sequence that looks like 6-8,10,12-13
    output = []
    for sec in s.split(","):
        items = [ int(i) for i in sec.split("-") ]
        if len(items) == 1:
            output.append(items[0])
        elif len(items) == 2:
            output.extend(range(items[0], items[1] + 1))
        else:
            raise RuntimeError(f"load_seq: invalid format: {s}")
    return output


def run_cmd(cmd, is_dry_run, async_cmd=True):
    if is_dry_run:
        print(f"[Cmd to be executed, async={async_cmd}] {cmd}")
    else:
        if async_cmd:
            Popen(cmd, preexec_fn=os.setpgrp, shell=True)
        else:
            check_output(cmd, shell=True)

def get_sacct(user_id):
    output = check_output(f"sacct -u {user_id} --json", shell=True)
    data = json.loads(output.decode("utf-8"))

    def _extract(submit_line, matcher):
        m = matcher.search(submit_line)
        return m.group(1) if m else None

    job_stats = dict()
    for entry in data["jobs"]:
        submit_line = entry["submit_line"]

        # Note that entry["comment"]["job"] is only nonempty when the job is not running. 
        # Therefore we cannot use it. Instead we extract directly from submit_line 
        jobid = entry["job_id"]
        comment = _extract(entry["submit_line"], comment_matcher)
        stderr_file = _extract(entry["submit_line"], stderr_matcher)
        stdout_file = _extract(entry["submit_line"], stdout_matcher)

        # replace any %j with the actual job id
        if stderr_file is not None:
            stderr_file = stderr_file.replace("%j", str(jobid)) 
        if stdout_file is not None:
            stdout_file = stdout_file.replace("%j", str(jobid)) 

        job_stats[jobid] = dict(
            state=entry["state"]["current"],
            comment=comment,
            stderr_file=stderr_file,
            stdout_file=stdout_file,
            submit_line=submit_line
        )

    return job_stats

def get_wandb_url(stderr_file):
    init_str = "wandb: 🚀 View run at "
    try:
        with open(stderr_file, "r") as f:
            for line in f:
                if line.startswith(init_str):
                    return line[len(init_str):].strip() 
    except:
        pass

    return None
