import re
from multiprocessing import Pool
import os
import sys
import json

s_matcher = re.compile(r"\s*([0-9]+:|)\s*(args|stats):")
def standard_matcher(prefix, f):
    arg_id = -1
    for line in f:
        m = s_matcher.match(line)
        if not m: continue

        task_id = m.group(1)
        t = m.group(2)
        if t == 'args':
            arg_id += 1
        key = "%s-%s-%d"% (prefix, task_id, arg_id)

        yield key, t, json.loads(line[len(m.group(0)):])

matcher = re.compile(r"Accumulated: [\d\.]+\[\d+\], Last episode\[16\] Avg: ([\d\.]+)")
matcher_step = re.compile(r"Total step: ([\d\.]+)M")

def simple_matcher(prefix, f):
    eval_mode = False
    train_mode = False
    for line in f:
        if line.startswith("Namespace"):
            args = eval(line.replace("Namespace", "dict"))
            yield prefix, "args", args
        elif line.startswith("Eval:"):
            eval_mode = True
        elif line.startswith("Train:"):
            train_mode = True
        elif train_mode:
            m = matcher_step.match(line)
            if m:
                last_step = float(m.group(1))
                train_mode = False
        elif eval_mode:
            m = matcher.match(line)
            if m:
                yield prefix, "stats", dict(step=last_step, test_avg=float(m.group(1)))
                eval_mode = False
                last_step = None

class ParallelParser:
    def parse_file(self, input):
        prefix, f = input
        data = {}
        if not os.path.exists(f):
            return data

        for key, t, content in self.match_iter(prefix, open(f)):
            if key not in data:
                data[key] = dict(stats=[], args=[])
            data[key][t].append(content)
        return data

    def parse(self, inputs):
        print(f"#input files = {len(inputs)}")
        p = Pool(64)
        data = p.map(self.parse_file, inputs)
        # data = [ parse_file(input) for input in inputs ]

        result = {}
        for d in data:
            result.update(d)
        return result

    def __init__(self, parser_name):
        self.match_iter = eval(parser_name)


