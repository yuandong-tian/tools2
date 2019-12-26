#!/home/yuandong/anaconda/bin
import argparse
import imp
import random
from collections import OrderedDict

def param_to_arg(param, env_vars=None):
    ''' Turn param to "--key value ", if value is bool, then
        it will switch on/off the key depending on the value
        env_vars exluded.
    '''
    arg = ""
    if isinstance(param, dict):
        param = [ (k, v) for k, v in param.items() ]
    for (k, v) in param:
        if env_vars is not None and k in env_vars: continue
        if k != "--":
            this_arg = " --%s" % k
            if not isinstance(v, bool):
                if isinstance(v, str) and (":" in v or ";" in v or "|" in v):
                    v = "\"" + v + "\""
                this_arg += " " + str(v)
                arg += this_arg
            else:
                if v: arg += this_arg
        else:
            arg += " --%s" % v
    return arg

def param_to_env(param, env_vars):
    if isinstance(env_vars, str):
        return env_vars
    else:
        if isinstance(param, dict):
            param = [ (k, v) for k, v in param.items() ]
        return " ".join("%s=%s" % (k, str(v)) for (k, v) in param if k in env_vars)


def iter_params(params):
    if isinstance(params, list):
        # List: just go through them one by one.
        for param in params:
            yield from iter_params(param)
    elif isinstance(params, dict):
        # remove the first key and yield on the rest
        if len(params) == 0:
            yield []
        else:
            # For each value of the first key,
            #   (key, value) + iter_params(other param)
            first_key = next(iter(params.keys()))
            dup_params = OrderedDict(params)
            del dup_params[first_key]

            if first_key.startswith("__sub"):
                for rest_instance in iter_params(dup_params):
                    for first_instance in iter_params(params[first_key]):
                        # print("__sub: First: " + str(first_instance))
                        # print("__sub: Rest: " + str(rest_instance))
                        yield first_instance + rest_instance
            else:
                for rest_instance in iter_params(dup_params):
                    # print("Rest_instance = " + str(rest_instance))
                    for first_val_instance in iter_params(params[first_key]):
                        # print("Key " + first_key + " val: " + str(first_val_instance))
                        yield [(first_key, first_val_instance)] + rest_instance
    else:
        yield params


def list2dict(l):
    return { p[0] : p[1] for p in l }


class ParamHandler:
    def __init__(self, param_file, num_per_group, param_args=None, shuffle=True):
        custom_params = imp.load_source("custom_params", param_file)
        if isinstance(custom_params.params, dict):
            params = custom_params.params
        else:
            params = custom_params.params(param_args)

        self.params = list(iter_params(params))
        if shuffle:
            random.shuffle(self.params)

        self.num_per_group = num_per_group if num_per_group > 0 else len(self.params) 
        self.exec_file = custom_params.main

    def get_num_jobs(self):
        return len(self.params)

    def get_num_groups(self):
        return (len(self.params) - 1) // self.num_per_group + 1

    def get_group(self, idx):
        if idx is None:
            return [ param_to_arg(param) for param in self.params ]
        else:
            offset = idx * self.num_per_group
            return [ param_to_arg(self.params[i]) for i in range(offset, min(offset + self.num_per_group, len(self.params))) ] 

    def get_exec(self, arg):
        return f"python -u {self.exec_file} {arg}"
            

# Load parameter files and generate args.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('param_file', type=str, help="Parameter file")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--num_per_group", type=int, default=1)
    parser.add_argument("--param_args", type=str, default=None)

    args = parser.parse_args()

    handler = ParamHandler(args.param_file, args.num_per_group, param_args=args.param_args)

    for arg in handler.get_group(args.idx):
        print(arg)
