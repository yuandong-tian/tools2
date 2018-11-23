from multiprocessing import Pool
import os
import glob
import json

def parse_file(input):
    import re
    import json

    task_matcher = re.compile(r"\s*([0-9]+:|)\s*(args|stats):")

    prefix, f = input
    data = {}
    arg_id = -1
    for line in open(f):
        m = task_matcher.match(line)
        if not m: continue
        task_id = m.group(1)
        t = m.group(2)
        if t == 'args':
            arg_id += 1
        key = "%s-%s-%d"% (prefix, task_id, arg_id)
        if key not in data:
            data[key] = dict(stats=[], args=[])
        data[key][t].append(json.loads(line[len(m.group(0)):]))
    return data

def parse_files(inputs):
    p = Pool(64)
    data = p.map(parse_file, inputs)
    # data = [ parse_file(input) for input in inputs ]

    result = {}
    for d in data:
        result.update(d)
    return result

def get(job_names):
    root = "/checkpoint/yuandong/jobs"
    inputs = []
    for job_name in job_names:
        for f in glob.glob(os.path.join(root, job_name, "*.out")):
            prefix = job_name + "-" + os.path.basename(f)
            inputs.append((prefix, f))

    return parse_files(inputs)

def get_aml(job_names):
    json_root = "/home/yuandong/tools/sweeper/jobs"
    root = "/mnt/vol/gfsai-flash-east/ai-group/users/yuandong/rts"

    inputs = []
    for job_name in job_names:
        with open(os.path.join(json_root, job_name + ".json")) as f:
            jobs = json.load(f)
        for job in jobs["jobs"]:
            if not "id" in job: continue
            f = "%d/output%d-0.log" % (job["id"], job["job_idx"])
            prefix = "%d-%d" % (job["id"], job["job_idx"])
            inputs.append((prefix, os.path.join(root, f)))

    return parse_files(inputs)

class AccuStats:
    def __init__(self):
        self.x = None
        self.xsqr = None
        self.counter = None

    def feed(self, v):
        l = v.shape[0]
        if self.x is None:
            self.x = v
            self.xsqr = v * v
            self.counter = np.ones(l)
        else:
            lmin = min(l, self.x.shape[0])
            self.x[:lmin] += v[:lmin]
            self.xsqr[:lmin] += v[:lmin] * v[:lmin]
            self.counter[:lmin] += 1

    def get_mean_std(self):
        mean = self.x / self.counter
        std = np.sqrt(self.xsqr / self.counter - mean ** 2)
        return mean, std


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_prefix', type=str, help="Json file")

    args = parser.parse_args()

    data = get_aml([args.json_prefix])
    print(data.keys())

