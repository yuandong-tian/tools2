from collections import defaultdict
import os
import re
import json
import glob

task_matcher = re.compile(r"\s*([0-9]+:|)\s(args|stats):")

def parse_file(prefix, f, data):
    arg_id = -1
    for line in open(f):
        m = task_matcher.match(line)
        if not m: continue
        task_id = m.group(1)
        t = m.group(2)
        if t == 'args':
            arg_id += 1
        key = "%s-%s-%d"% (prefix, task_id, arg_id)
        data[key][t].append(json.loads(line[len(m.group(0)):]))

def get(job_names):
    root = "/checkpoint/yuandong/jobs"
    data = defaultdict(lambda: dict(args=[], stats=[]))
    for job_name in job_names:
        for f in glob.glob(os.path.join(root, job_name, "*.out")):
            prefix = job_name + "-" + os.path.basename(f)
            parse_file(prefix, f, data)
    return data

def get_aml(job_names):
    json_root = "/home/yuandong/tools/sweeper/jobs"
    root = "/mnt/vol/gfsai-flash-east/ai-group/users/yuandong/rts"
    
    data = defaultdict(lambda: dict(args=[], stats=[]))
    for job_name in job_names:
        with open(os.path.join(json_root, job_name + ".json")) as f:
            data = json.load(f)
        for job in data["jobs"]:
            if not "id" in job: continue
            f = "%d/output%d-0.log" % (job["id"], job["job_idx"])
            parse_file(root, f, data)

    return data
            
